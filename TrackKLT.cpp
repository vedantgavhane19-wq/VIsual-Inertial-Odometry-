/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "TrackKLT.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>

#include "Grider_FAST.h"
#include "Grider_GRID.h"
#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"

using namespace ov_core;

TrackKLT::TrackKLT(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                   HistogramMethod histmethod, int fast_threshold, int gridx, int gridy, int minpxdist)
    : TrackBase(cameras, numfeats, numaruco, stereo, histmethod), threshold(fast_threshold), grid_x(gridx), grid_y(gridy),
      min_px_dist(minpxdist) {
  init_fpga();
}

TrackKLT::~TrackKLT() {
  close_fpga();
}

void TrackKLT::init_fpga() {
  // 1. Read physical base address dynamically from udmabuf0 sysfs
  FILE* f_addr = fopen("/sys/class/u-dma-buf/udmabuf0/phys_addr", "r");
  if (!f_addr || fscanf(f_addr, "%lx", &fpga_phys_addr) != 1) {
    PRINT_WARNING(YELLOW "[OpenVINS-FPGA] Warning: Cannot read /sys/class/u-dma-buf/udmabuf0/phys_addr. Falling back to CPU FAST.\n" RESET);
    if (f_addr) fclose(f_addr);
    fpga_initialized = false;
    return;
  }
  fclose(f_addr);

  // 2. Setup udmabuf cache sync parameters
  FILE* f_sz = fopen("/sys/class/u-dma-buf/udmabuf0/sync_size", "w");
  if (f_sz) {
    fprintf(f_sz, "%zu\n", FPGA_TOTAL_BYTES);
    fclose(f_sz);
  }
  FILE* f_dir = fopen("/sys/class/u-dma-buf/udmabuf0/sync_direction", "w");
  if (f_dir) {
    fprintf(f_dir, "3\n"); // 3 = DMA_BIDIRECTIONAL
    fclose(f_dir);
  }
  FILE* f_off = fopen("/sys/class/u-dma-buf/udmabuf0/sync_offset", "w");
  if (f_off) {
    fprintf(f_off, "0\n");
    fclose(f_off);
  }

  // 3. Open udmabuf0 for DMA input and output buffers
  fpga_udma_fd = open("/dev/udmabuf0", O_RDWR);
  if (fpga_udma_fd < 0) {
    PRINT_WARNING(YELLOW "[OpenVINS-FPGA] Warning: Failed to open /dev/udmabuf0. Falling back to CPU FAST.\n" RESET);
    fpga_initialized = false;
    return;
  }
  fpga_shared_mem = (uint8_t*)mmap(NULL, FPGA_TOTAL_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fpga_udma_fd, 0);
  if (fpga_shared_mem == MAP_FAILED) {
    PRINT_WARNING(YELLOW "[OpenVINS-FPGA] Warning: Failed to mmap udmabuf0. Falling back to CPU FAST.\n" RESET);
    close(fpga_udma_fd);
    fpga_udma_fd = -1;
    fpga_shared_mem = nullptr;
    fpga_initialized = false;
    return;
  }

  // 4. Open /dev/mem for AXI control registers
  fpga_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fpga_mem_fd < 0) {
    PRINT_WARNING(YELLOW "[OpenVINS-FPGA] Warning: Failed to open /dev/mem (run ROS 2 with sudo). Falling back to CPU FAST.\n" RESET);
    munmap(fpga_shared_mem, FPGA_TOTAL_BYTES);
    close(fpga_udma_fd);
    fpga_shared_mem = nullptr;
    fpga_udma_fd = -1;
    fpga_initialized = false;
    return;
  }
  fpga_axi = (volatile uint32_t*)mmap(NULL, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, fpga_mem_fd, 0xB0000000);
  if (fpga_axi == MAP_FAILED) {
    PRINT_WARNING(YELLOW "[OpenVINS-FPGA] Warning: Failed to mmap AXI control registers. Falling back to CPU FAST.\n" RESET);
    close(fpga_mem_fd);
    munmap(fpga_shared_mem, FPGA_TOTAL_BYTES);
    close(fpga_udma_fd);
    fpga_axi = nullptr;
    fpga_mem_fd = -1;
    fpga_shared_mem = nullptr;
    fpga_udma_fd = -1;
    fpga_initialized = false;
    return;
  }

  // 5. Open sync sysfs file descriptors
  fpga_sync_dev = open("/sys/class/u-dma-buf/udmabuf0/sync_for_device", O_WRONLY);
  fpga_sync_cpu = open("/sys/class/u-dma-buf/udmabuf0/sync_for_cpu", O_WRONLY);

  // 6. Write static AXI configuration (matching verified PYNQ / C++ driver)
  uint64_t phys_out = fpga_phys_addr + FPGA_FRAME_BYTES;
  fpga_axi[0x10 / 4] = (uint32_t)(fpga_phys_addr & 0xFFFFFFFF);         // SRC_LO
  fpga_axi[0x14 / 4] = (uint32_t)((fpga_phys_addr >> 32) & 0xFFFFFFFF); // SRC_HI
  fpga_axi[0x1C / 4] = (uint32_t)(phys_out & 0xFFFFFFFF);              // DST_LO
  fpga_axi[0x20 / 4] = (uint32_t)((phys_out >> 32) & 0xFFFFFFFF);      // DST_HI
  fpga_axi[0x28 / 4] = FPGA_HEIGHT;                                     // ROWS (320)
  fpga_axi[0x30 / 4] = FPGA_WIDTH;                                      // COLS (480)
  fpga_axi[0x38 / 4] = FPGA_THRESHOLD;                                  // THRESH (1500)
  fpga_axi[0x40 / 4] = FPGA_K_SCALED;                                   // K_SCALE (2621)

  fpga_initialized = true;
  PRINT_INFO(GREEN "[OpenVINS-FPGA] Harris Corner Detector Accelerator initialized at 0xB0000000!\n" RESET);
}

void TrackKLT::close_fpga() {
  if (fpga_sync_dev >= 0) { close(fpga_sync_dev); fpga_sync_dev = -1; }
  if (fpga_sync_cpu >= 0) { close(fpga_sync_cpu); fpga_sync_cpu = -1; }
  if (fpga_axi && fpga_axi != MAP_FAILED) { munmap((void*)fpga_axi, 0x10000); fpga_axi = nullptr; }
  if (fpga_mem_fd >= 0) { close(fpga_mem_fd); fpga_mem_fd = -1; }
  if (fpga_shared_mem && fpga_shared_mem != MAP_FAILED) { munmap(fpga_shared_mem, FPGA_TOTAL_BYTES); fpga_shared_mem = nullptr; }
  if (fpga_udma_fd >= 0) { close(fpga_udma_fd); fpga_udma_fd = -1; }
  fpga_initialized = false;
}

void TrackKLT::detect_fpga(const cv::Mat &img, std::vector<cv::KeyPoint> &pts_ext) {
  if (!fpga_initialized || img.empty()) return;

  std::lock_guard<std::mutex> lck(mtx_fpga);

  uint8_t *in_buf = fpga_shared_mem;
  uint8_t *out_buf = fpga_shared_mem + FPGA_FRAME_BYTES;

  // 1. Prepare 480x320 grayscale frame for hardware
  cv::Mat gray_in;
  if (img.channels() == 3) {
    cv::cvtColor(img, gray_in, cv::COLOR_BGR2GRAY);
  } else {
    gray_in = img;
  }

  cv::Mat img_fpga;
  if (gray_in.cols != FPGA_WIDTH || gray_in.rows != FPGA_HEIGHT) {
    cv::resize(gray_in, img_fpga, cv::Size(FPGA_WIDTH, FPGA_HEIGHT));
  } else {
    img_fpga = gray_in;
  }

  // 2. Transfer frame to DMA buffer and clear output
  memcpy(in_buf, img_fpga.data, FPGA_FRAME_BYTES);
  memset(out_buf, 0, FPGA_FRAME_BYTES);

  // 3. Flush CPU cache to DDR for FPGA
  if (fpga_sync_dev >= 0) {
    lseek(fpga_sync_dev, 0, SEEK_SET);
    ssize_t s = write(fpga_sync_dev, "1", 1);
    (void)s;
  }

  // 4. Trigger FPGA execution
  fpga_axi[0x00 / 4] = 0x01; // AP_START

  // Hardware busy-wait polling (~1.1 ms)
  uint32_t timeout = 0;
  while (!(fpga_axi[0x00 / 4] & 0x02)) {
    timeout++;
    if (timeout > 50000000) {
      PRINT_ERROR(RED "[OpenVINS-FPGA] Timeout! Hardware deadlocked. AXI Control: 0x%X\n" RESET, fpga_axi[0x00 / 4]);
      fpga_axi[0x00 / 4] = 0x00;
      return;
    }
  }
  fpga_axi[0x00 / 4] = 0x00;

  // 5. Invalidate CPU cache to read FPGA results from DDR
  if (fpga_sync_cpu >= 0) {
    lseek(fpga_sync_cpu, 0, SEEK_SET);
    ssize_t s = write(fpga_sync_cpu, "1", 1);
    (void)s;
  }

  // 6. Connected components centroid extraction
  cv::Mat mask_fpga(FPGA_HEIGHT, FPGA_WIDTH, CV_8UC1, out_buf);
  cv::Mat binary_mask = (mask_fpga > 0);
  cv::Mat labels, stats, centroids;
  int n_labels = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8);

  float scale_u = (float)img.cols / (float)FPGA_WIDTH;
  float scale_v = (float)img.rows / (float)FPGA_HEIGHT;

  for (int i = 1; i < n_labels && i <= 1000; i++) {
    float cx = static_cast<float>(centroids.at<double>(i, 0)) * scale_u;
    float cy = static_cast<float>(centroids.at<double>(i, 1)) * scale_v;
    pts_ext.emplace_back(cx, cy, 1.0f);
  }
}

void TrackKLT::feed_new_camera(const CameraData &message) {

  // Error check that we have all the data
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
    PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
    std::exit(EXIT_FAILURE);
  }

  rT1 = boost::posix_time::microsec_clock::local_time();
  size_t num_images = message.images.size();
  for (size_t msg_id = 0; msg_id < num_images; msg_id++) {

    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    cv::Mat img;
    if (histogram_method == HistogramMethod::HISTOGRAM) {
      cv::equalizeHist(message.images.at(msg_id), img);
    } else if (histogram_method == HistogramMethod::CLAHE) {
      double eq_clip_limit = 10.0;
      cv::Size eq_win_size = cv::Size(8, 8);
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
      clahe->apply(message.images.at(msg_id), img);
    } else {
      img = message.images.at(msg_id);
    }

    std::vector<cv::Mat> imgpyr;
    cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);

    img_curr[cam_id] = img;
    img_pyramid_curr[cam_id] = imgpyr;
  }

  if (num_images == 1) {
    feed_monocular(message, 0);
  } else if (num_images == 2 && use_stereo) {
    feed_stereo(message, 0, 1);
  } else if (!use_stereo) {
    parallel_for_(cv::Range(0, (int)num_images), LambdaBody([&](const cv::Range &range) {
                    for (int i = range.start; i < range.end; i++) {
                      feed_monocular(message, i);
                    }
                  }));
  } else {
    PRINT_ERROR(RED "[ERROR]: invalid number of images passed %zu, we only support mono or stereo tracking", num_images);
    std::exit(EXIT_FAILURE);
  }
}

void TrackKLT::feed_monocular(const CameraData &message, size_t msg_id) {

  size_t cam_id = message.sensor_ids.at(msg_id);
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

  cv::Mat img = img_curr.at(cam_id);
  std::vector<cv::Mat> imgpyr = img_pyramid_curr.at(cam_id);
  cv::Mat mask = message.masks.at(msg_id);
  rT2 = boost::posix_time::microsec_clock::local_time();

  if (pts_last[cam_id].empty()) {
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;
    perform_detection_monocular(imgpyr, mask, good_left, good_ids_left);

    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
    return;
  }

  int pts_before_detect = (int)pts_last[cam_id].size();
  auto pts_left_old = pts_last[cam_id];
  auto ids_left_old = ids_last[cam_id];
  perform_detection_monocular(img_pyramid_last[cam_id], img_mask_last[cam_id], pts_left_old, ids_left_old);
  rT3 = boost::posix_time::microsec_clock::local_time();

  std::vector<uchar> mask_ll;
  std::vector<cv::KeyPoint> pts_left_new = pts_left_old;

  perform_matching(img_pyramid_last[cam_id], imgpyr, pts_left_old, pts_left_new, cam_id, cam_id, mask_ll);
  assert(pts_left_new.size() == ids_left_old.size());
  rT4 = boost::posix_time::microsec_clock::local_time();

  if (mask_ll.empty()) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id].clear();
    ids_last[cam_id].clear();
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  std::vector<cv::KeyPoint> good_left;
  std::vector<size_t> good_ids_left;

  for (size_t i = 0; i < pts_left_new.size(); i++) {
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= img.cols ||
        (int)pts_left_new.at(i).pt.y >= img.rows)
      continue;

    if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
      continue;

    if (mask_ll[i]) {
      good_left.push_back(pts_left_new[i]);
      good_ids_left.push_back(ids_left_old[i]);
    }
  }

  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x, npt_l.y);
  }

  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
  }
  rT5 = boost::posix_time::microsec_clock::local_time();

  PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%zu detected)\n", (rT3 - rT2).total_microseconds() * 1e-6,
            (int)pts_last[cam_id].size() - pts_before_detect);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", (rT5 - rT4).total_microseconds() * 1e-6,
            (int)good_left.size());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}

void TrackKLT::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right) {

  size_t cam_id_left = message.sensor_ids.at(msg_id_left);
  size_t cam_id_right = message.sensor_ids.at(msg_id_right);
  std::lock_guard<std::mutex> lck1(mtx_feeds.at(cam_id_left));
  std::lock_guard<std::mutex> lck2(mtx_feeds.at(cam_id_right));

  cv::Mat img_left = img_curr.at(cam_id_left);
  cv::Mat img_right = img_curr.at(cam_id_right);
  std::vector<cv::Mat> imgpyr_left = img_pyramid_curr.at(cam_id_left);
  std::vector<cv::Mat> imgpyr_right = img_pyramid_curr.at(cam_id_right);
  cv::Mat mask_left = message.masks.at(msg_id_left);
  cv::Mat mask_right = message.masks.at(msg_id_right);
  rT2 = boost::posix_time::microsec_clock::local_time();

  if (pts_last[cam_id_left].empty() && pts_last[cam_id_right].empty()) {
    std::vector<cv::KeyPoint> good_left, good_right;
    std::vector<size_t> good_ids_left, good_ids_right;
    perform_detection_stereo(imgpyr_left, imgpyr_right, mask_left, mask_right, cam_id_left, cam_id_right, good_left, good_right,
                             good_ids_left, good_ids_right);

    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
    return;
  }

  int pts_before_detect = (int)pts_last[cam_id_left].size();
  auto pts_left_old = pts_last[cam_id_left];
  auto pts_right_old = pts_last[cam_id_right];
  auto ids_left_old = ids_last[cam_id_left];
  auto ids_right_old = ids_last[cam_id_right];
  perform_detection_stereo(img_pyramid_last[cam_id_left], img_pyramid_last[cam_id_right], img_mask_last[cam_id_left],
                           img_mask_last[cam_id_right], cam_id_left, cam_id_right, pts_left_old, pts_right_old, ids_left_old,
                           ids_right_old);
  rT3 = boost::posix_time::microsec_clock::local_time();

  std::vector<uchar> mask_ll, mask_rr;
  std::vector<cv::KeyPoint> pts_left_new = pts_left_old;
  std::vector<cv::KeyPoint> pts_right_new = pts_right_old;

  parallel_for_(cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
                  for (int i = range.start; i < range.end; i++) {
                    bool is_left = (i == 0);
                    perform_matching(img_pyramid_last[is_left ? cam_id_left : cam_id_right], is_left ? imgpyr_left : imgpyr_right,
                                     is_left ? pts_left_old : pts_right_old, is_left ? pts_left_new : pts_right_new,
                                     is_left ? cam_id_left : cam_id_right, is_left ? cam_id_left : cam_id_right,
                                     is_left ? mask_ll : mask_rr);
                  }
                }));
  rT4 = boost::posix_time::microsec_clock::local_time();
  rT5 = boost::posix_time::microsec_clock::local_time();

  if (mask_ll.empty() && mask_rr.empty()) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left].clear();
    pts_last[cam_id_right].clear();
    ids_last[cam_id_left].clear();
    ids_last[cam_id_right].clear();
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  std::vector<cv::KeyPoint> good_left, good_right;
  std::vector<size_t> good_ids_left, good_ids_right;

  for (size_t i = 0; i < pts_left_new.size(); i++) {
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x > img_left.cols ||
        (int)pts_left_new.at(i).pt.y > img_left.rows)
      continue;

    bool found_right = false;
    size_t index_right = 0;
    for (size_t n = 0; n < ids_right_old.size(); n++) {
      if (ids_left_old.at(i) == ids_right_old.at(n)) {
        found_right = true;
        index_right = n;
        break;
      }
    }

    if (mask_ll[i] && found_right && mask_rr[index_right]) {
      if (pts_right_new.at(index_right).pt.x < 0 || pts_right_new.at(index_right).pt.y < 0 ||
          (int)pts_right_new.at(index_right).pt.x >= img_right.cols || (int)pts_right_new.at(index_right).pt.y >= img_right.rows)
        continue;
      good_left.push_back(pts_left_new[i]);
      good_right.push_back(pts_right_new[index_right]);
      good_ids_left.push_back(ids_left_old[i]);
      good_ids_right.push_back(ids_right_old[index_right]);
    } else if (mask_ll[i]) {
      good_left.push_back(pts_left_new[i]);
      good_ids_left.push_back(ids_left_old[i]);
    }
  }

  for (size_t i = 0; i < pts_right_new.size(); i++) {
    if (pts_right_new.at(i).pt.x < 0 || pts_right_new.at(i).pt.y < 0 || (int)pts_right_new.at(i).pt.x >= img_right.cols ||
        (int)pts_right_new.at(i).pt.y >= img_right.rows)
      continue;

    bool added_already = (std::find(good_ids_right.begin(), good_ids_right.end(), ids_right_old.at(i)) != good_ids_right.end());
    if (mask_rr[i] && !added_already) {
      good_right.push_back(pts_right_new[i]);
      good_ids_right.push_back(ids_right_old[i]);
    }
  }

  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                             npt_l.y);
  }
  for (size_t i = 0; i < good_right.size(); i++) {
    cv::Point2f npt_r = camera_calib.at(cam_id_right)->undistort_cv(good_right.at(i).pt);
    database->update_feature(good_ids_right.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                             npt_r.y);
  }

  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
  }
  rT6 = boost::posix_time::microsec_clock::local_time();

  PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%d detected)\n", (rT3 - rT2).total_microseconds() * 1e-6,
            (int)pts_last[cam_id_left].size() - pts_before_detect);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for stereo klt\n", (rT5 - rT4).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", (rT6 - rT5).total_microseconds() * 1e-6,
            (int)good_left.size());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", (rT6 - rT1).total_microseconds() * 1e-6);
}

void TrackKLT::perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                           std::vector<size_t> &ids0) {

  cv::Size size_close((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                      (int)((float)img0pyr.at(0).rows / (float)min_px_dist));
  cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
  float size_x = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid(grid_x, grid_y);
  cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();

  while (it0 != pts0.end()) {
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    int x_grid = std::floor(kpt.pt.x / size_x);
    int y_grid = std::floor(kpt.pt.y / size_y);
    if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
    }

    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  double min_feat_percent = 0.50;
  int num_featsneeded = num_features - (int)pts0.size();
  if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features)))
    return;

  // Grid mask calculation for fallback
  cv::Mat mask0_grid;
  cv::resize(mask0, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_NEAREST);
  int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
  int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
  std::vector<std::pair<int, int>> valid_locs;
  for (int x = 0; x < grid_2d_grid.cols; x++) {
    for (int y = 0; y < grid_2d_grid.rows; y++) {
      if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
        valid_locs.emplace_back(x, y);
      }
    }
  }

  // --- FEATURE EXTRACTION (FPGA ACCELERATED WITH CPU FALLBACK) ---
  std::vector<cv::KeyPoint> pts0_ext;
  if (fpga_initialized) {
    detect_fpga(img0pyr.at(0), pts0_ext);
  } else {
    Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);
  }

  // Filter detected points against occupancy grid and exclusion masks
  std::vector<cv::KeyPoint> kpts0_new;
  std::vector<cv::Point2f> pts0_new;
  for (auto &kpt : pts0_ext) {
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
      continue;

    if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
      continue;

    int px = (int)kpt.pt.x;
    int py = (int)kpt.pt.y;
    if (px >= 0 && px < mask0.cols && py >= 0 && py < mask0.rows && mask0.at<uint8_t>(py, px) > 127)
      continue;

    kpts0_new.push_back(kpt);
    pts0_new.push_back(kpt.pt);
    grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
  }

  // Append new features up to configured num_features limit
  for (size_t i = 0; i < pts0_new.size(); i++) {
    if ((int)pts0.size() >= num_features)
      break;

    kpts0_new.at(i).pt = pts0_new.at(i);
    pts0.push_back(kpts0_new.at(i));
    size_t temp = ++currid;
    ids0.push_back(temp);
  }
}

void TrackKLT::perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                        const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                        std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1) {

  cv::Size size_close0((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                       (int)((float)img0pyr.at(0).rows / (float)min_px_dist));
  cv::Mat grid_2d_close0 = cv::Mat::zeros(size_close0, CV_8UC1);
  float size_x0 = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y0 = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid0(grid_x, grid_y);
  cv::Mat grid_2d_grid0 = cv::Mat::zeros(size_grid0, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();

  while (it0 != pts0.end()) {
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close0.width || y_close < 0 || y_close >= size_close0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    int x_grid = std::floor(kpt.pt.x / size_x0);
    int y_grid = std::floor(kpt.pt.y / size_y0);
    if (x_grid < 0 || x_grid >= size_grid0.width || y_grid < 0 || y_grid >= size_grid0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    if (grid_2d_close0.at<uint8_t>(y_close, x_close) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    grid_2d_close0.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid0.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid0.at<uint8_t>(y_grid, x_grid) += 1;
    }

    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  double min_feat_percent = 0.50;
  int num_featsneeded_0 = num_features - (int)pts0.size();

  if (num_featsneeded_0 > std::min(20, (int)(min_feat_percent * num_features))) {

    cv::Mat mask0_grid;
    cv::resize(mask0, mask0_grid, size_grid0, 0.0, 0.0, cv::INTER_NEAREST);
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid0.cols; x++) {
      for (int y = 0; y < grid_2d_grid0.rows; y++) {
        if ((int)grid_2d_grid0.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }

    std::vector<cv::KeyPoint> pts0_ext;
    if (fpga_initialized) {
      detect_fpga(img0pyr.at(0), pts0_ext);
    } else {
      Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);
    }

    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    for (auto &kpt : pts0_ext) {
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close0.width || y_grid < 0 || y_grid >= size_close0.height)
        continue;

      if (grid_2d_close0.at<uint8_t>(y_grid, x_grid) > 127)
        continue;

      grid_2d_close0.at<uint8_t>(y_grid, x_grid) = 255;
      kpts0_new.push_back(kpt);
      pts0_new.push_back(kpt.pt);
    }

    std::vector<cv::KeyPoint> kpts1_new = kpts0_new;
    std::vector<cv::Point2f> pts1_new = pts0_new;

    if (!pts0_new.empty()) {
      std::vector<uchar> mask;
      std::vector<float> error;
      cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
      cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0_new, pts1_new, mask, error, win_size, pyr_levels, term_crit,
                               cv::OPTFLOW_USE_INITIAL_FLOW);

      for (size_t i = 0; i < pts0_new.size(); i++) {
        if ((int)pts0.size() >= num_features)
          break;

        bool oob_left = ((int)pts0_new.at(i).x < 0 || (int)pts0_new.at(i).x >= img0pyr.at(0).cols || (int)pts0_new.at(i).y < 0 ||
                         (int)pts0_new.at(i).y >= img0pyr.at(0).rows);
        bool oob_right = ((int)pts1_new.at(i).x < 0 || (int)pts1_new.at(i).x >= img1pyr.at(0).cols || (int)pts1_new.at(i).y < 0 ||
                          (int)pts1_new.at(i).y >= img1pyr.at(0).rows);

        if (!oob_left && !oob_right && mask[i] == 1) {
          kpts0_new.at(i).pt = pts0_new.at(i);
          kpts1_new.at(i).pt = pts1_new.at(i);
          pts0.push_back(kpts0_new.at(i));
          pts1.push_back(kpts1_new.at(i));
          size_t temp = ++currid;
          ids0.push_back(temp);
          ids1.push_back(temp);
        } else if (!oob_left) {
          kpts0_new.at(i).pt = pts0_new.at(i);
          pts0.push_back(kpts0_new.at(i));
          size_t temp = ++currid;
          ids0.push_back(temp);
        }
      }
    }
  }

  // Right camera extraction
  cv::Size size_close1((int)((float)img1pyr.at(0).cols / (float)min_px_dist), (int)((float)img1pyr.at(0).rows / (float)min_px_dist));
  cv::Mat grid_2d_close1 = cv::Mat::zeros(size_close1, CV_8UC1);
  float size_x1 = (float)img1pyr.at(0).cols / (float)grid_x;
  float size_y1 = (float)img1pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid1(grid_x, grid_y);
  cv::Mat grid_2d_grid1 = cv::Mat::zeros(size_grid1, CV_8UC1);
  cv::Mat mask1_updated = mask0.clone();
  it0 = pts1.begin();
  it1 = ids1.begin();

  while (it0 != pts1.end()) {
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img1pyr.at(0).cols - edge || y < edge || y >= img1pyr.at(0).rows - edge) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }

    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close1.width || y_close < 0 || y_close >= size_close1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }

    int x_grid = std::floor(kpt.pt.x / size_x1);
    int y_grid = std::floor(kpt.pt.y / size_y1);
    if (x_grid < 0 || x_grid >= size_grid1.width || y_grid < 0 || y_grid >= size_grid1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }

    bool is_stereo = (std::find(ids0.begin(), ids0.end(), *it1) != ids0.end());
    if (grid_2d_close1.at<uint8_t>(y_close, x_close) > 127 && !is_stereo) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }

    if (mask1.at<uint8_t>(y, x) > 127) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }

    grid_2d_close1.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid1.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid1.at<uint8_t>(y_grid, x_grid) += 1;
    }

    if (x - min_px_dist >= 0 && x + min_px_dist < img1pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img1pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask1_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  int num_featsneeded_1 = num_features - (int)pts1.size();
  if (num_featsneeded_1 > std::min(20, (int)(min_feat_percent * num_features))) {

    cv::Mat mask1_grid;
    cv::resize(mask1, mask1_grid, size_grid1, 0.0, 0.0, cv::INTER_NEAREST);
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid1.cols; x++) {
      for (int y = 0; y < grid_2d_grid1.rows; y++) {
        if ((int)grid_2d_grid1.at<uint8_t>(y, x) < num_features_grid_req && (int)mask1_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }

    std::vector<cv::KeyPoint> pts1_ext;
    if (fpga_initialized) {
      detect_fpga(img1pyr.at(0), pts1_ext);
    } else {
      Grider_GRID::perform_griding(img1pyr.at(0), mask1_updated, valid_locs, pts1_ext, num_features, grid_x, grid_y, threshold, true);
    }

    for (auto &kpt : pts1_ext) {
      if ((int)pts1.size() >= num_features)
        break;

      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close1.width || y_grid < 0 || y_grid >= size_close1.height)
        continue;

      if (grid_2d_close1.at<uint8_t>(y_grid, x_grid) > 127)
        continue;

      pts1.push_back(kpt);
      size_t temp = ++currid;
      ids1.push_back(temp);
      grid_2d_close1.at<uint8_t>(y_grid, x_grid) = 255;
    }
  }
}

void TrackKLT::perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &kpts0,
                                std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1, std::vector<uchar> &mask_out) {

  assert(kpts0.size() == kpts1.size());
  if (kpts0.empty() || kpts1.empty())
    return;

  std::vector<cv::Point2f> pts0, pts1;
  for (size_t i = 0; i < kpts0.size(); i++) {
    pts0.push_back(kpts0.at(i).pt);
    pts1.push_back(kpts1.at(i).pt);
  }

  if (pts0.size() < 10) {
    for (size_t i = 0; i < pts0.size(); i++)
      mask_out.push_back((uchar)0);
    return;
  }

  std::vector<uchar> mask_klt;
  std::vector<float> error;
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);

  std::vector<cv::Point2f> pts0_n, pts1_n;
  for (size_t i = 0; i < pts0.size(); i++) {
    pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0.at(i)));
    pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1.at(i)));
  }

  std::vector<uchar> mask_rsc;
  double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
  double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
  double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
  cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_rsc);

  for (size_t i = 0; i < mask_klt.size(); i++) {
    auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0);
    mask_out.push_back(mask);
  }

  for (size_t i = 0; i < pts0.size(); i++) {
    kpts0.at(i).pt = pts0.at(i);
    kpts1.at(i).pt = pts1.at(i);
  }
}
