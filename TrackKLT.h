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

#ifndef OV_CORE_TRACK_KLT_H
#define OV_CORE_TRACK_KLT_H

#include "TrackBase.h"
#include <mutex>

namespace ov_core {

/**
 * @brief KLT tracking of features.
 *
 * This is the implementation of a KLT visual frontend for tracking sparse features.
 * We can track either monocular cameras across time (temporally) along with
 * stereo cameras which we also track across time (temporally) but track from left to right
 * to find the stereo correspondence information also.
 * This uses the calcOpticalFlowPyrLK OpenCV function to do the KLT tracking.
 */
class TrackKLT : public TrackBase {

public:
  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in it
   * @param numfeats number of features we want to track
   * @param numaruco max id of arucotags
   * @param stereo if we should do stereo feature tracking or binocular
   * @param histmethod what type of histogram pre-processing should be done
   * @param fast_threshold FAST detection threshold
   * @param gridx size of grid in x-direction
   * @param gridy size of grid in y-direction
   * @param minpxdist features need to be at least this number pixels away from each other
   */
  explicit TrackKLT(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                    HistogramMethod histmethod, int fast_threshold, int gridx, int gridy, int minpxdist);

  ~TrackKLT() override;

  /**
   * @brief Process a new image
   * @param message Contains timestamp, images, and camera ids
   */
  void feed_new_camera(const CameraData &message) override;

protected:
  /**
   * @brief Process a new monocular image
   * @param message Contains timestamp, images, and camera ids
   * @param msg_id camera index in message data vector
   */
  void feed_monocular(const CameraData &message, size_t msg_id);

  /**
   * @brief Process new stereo pair of images
   * @param message Contains timestamp, images, and camera ids
   * @param msg_id_left first image index in message data vector
   * @param msg_id_right second image index in message data vector
   */
  void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);

  /**
   * @brief Detects new features in the current image
   * @param img0pyr image pyramid (level 0 is base)
   * @param mask0 mask representing ROI to exclude
   * @param pts0 vector of currently extracted keypoints
   * @param ids0 vector of feature ids for extracted keypoints
   */
  void perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                   std::vector<size_t> &ids0);

  /**
   * @brief Detects new features in the current stereo pair
   */
  void perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1);

  /**
   * @brief KLT track between two images, and run RANSAC outlier rejection
   */
  void perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                        std::vector<cv::KeyPoint> &pts1, size_t id0, size_t id1, std::vector<uchar> &mask_out);

  // Parameters for FAST grid detector
  int threshold;
  int grid_x;
  int grid_y;

  // Minimum pixel distance to be far away enough to be an extracted feature
  int min_px_dist;

  // How many pyramid levels to track
  int pyr_levels = 5;
  cv::Size win_size = cv::Size(15, 15);

  // Last set of image pyramids
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;
  std::map<size_t, cv::Mat> img_curr;
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_curr;

  // =========================================================================
  // Hardware FPGA Acceleration Members
  // =========================================================================
  bool fpga_initialized = false;
  int fpga_mem_fd = -1;
  int fpga_udma_fd = -1;
  int fpga_sync_dev = -1;
  int fpga_sync_cpu = -1;
  uint8_t *fpga_shared_mem = nullptr;
  volatile uint32_t *fpga_axi = nullptr;
  uint64_t fpga_phys_addr = 0;
  std::mutex mtx_fpga;

  static constexpr int FPGA_WIDTH = 480;
  static constexpr int FPGA_HEIGHT = 320;
  static constexpr size_t FPGA_FRAME_BYTES = 480 * 320;
  static constexpr size_t FPGA_TOTAL_BYTES = 480 * 320 * 2;
  static constexpr int FPGA_THRESHOLD = 1500;
  static constexpr int FPGA_K_SCALED = 2621;

  void init_fpga();
  void close_fpga();
  void detect_fpga(const cv::Mat &img, std::vector<cv::KeyPoint> &pts_ext);
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_KLT_H */
