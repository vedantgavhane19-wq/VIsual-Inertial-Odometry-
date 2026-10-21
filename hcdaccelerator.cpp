#ifndef HCD_ACCELERATOR_HPP
#define HCD_ACCELERATOR_HPP

#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>

class HcdAccelerator {
public:
    static constexpr int WIDTH = 480;
    static constexpr int HEIGHT = 320;
    static constexpr size_t FRAME_BYTES = WIDTH * HEIGHT;
    static constexpr size_t TOTAL_BYTES = FRAME_BYTES * 2;
    static constexpr int THRESHOLD = 1500;
    static constexpr int K_SCALED = 2621;

    HcdAccelerator() : is_valid_(false), udma_fd_(-1), mem_fd_(-1), sync_dev_(-1), sync_cpu_(-1),
                       shared_mem_(nullptr), axi_(nullptr) {
        init_hardware();
    }

    ~HcdAccelerator() {
        if (sync_dev_ >= 0) close(sync_dev_);
        if (sync_cpu_ >= 0) close(sync_cpu_);
        if (axi_) munmap((void*)axi_, 0x10000);
        if (mem_fd_ >= 0) close(mem_fd_);
        if (shared_mem_) munmap(shared_mem_, TOTAL_BYTES);
        if (udma_fd_ >= 0) close(udma_fd_);
    }

    bool is_ready() const { return is_valid_; }

    // Run FPGA detection and return corners scaled to input image resolution
    void detect(const cv::Mat& input_img, std::vector<cv::Point2f>& pts, int max_features = 800) {
        pts.clear();
        if (!is_valid_ || input_img.empty()) return;

        // 1. Prepare grayscale image resized to FPGA resolution
        cv::Mat gray_resized;
        if (input_img.channels() == 3) {
            cv::cvtColor(input_img, gray_resized, cv::COLOR_BGR2GRAY);
        } else {
            gray_resized = input_img;
        }

        bool need_scale = (gray_resized.cols != WIDTH || gray_resized.rows != HEIGHT);
        float scale_x = (float)input_img.cols / WIDTH;
        float scale_y = (float)input_img.rows / HEIGHT;

        if (need_scale) {
            cv::resize(gray_resized, gray_resized, cv::Size(WIDTH, HEIGHT));
        }

        // 2. DMA transfer and cache flush
        memcpy(shared_mem_, gray_resized.data, FRAME_BYTES);

        if (sync_dev_ >= 0) {
            lseek(sync_dev_, 0, SEEK_SET);
            ssize_t s = write(sync_dev_, "1", 1);
            (void)s;
        }

        // 3. Trigger FPGA
        axi_[0x00 / 4] = 0x01; // AP_START

        // Busy-wait polling (~1.1 ms execution)
        uint32_t timeout = 0;
        while (!(axi_[0x00 / 4] & 0x02)) {
            if (++timeout > 5000000) {
                std::cerr << "[OpenVINS-FPGA] Timeout! Hardware stall.\n";
                axi_[0x00 / 4] = 0x00;
                return;
            }
        }
        axi_[0x00 / 4] = 0x00;

        // 4. Invalidate CPU cache
        if (sync_cpu_ >= 0) {
            lseek(sync_cpu_, 0, SEEK_SET);
            ssize_t s = write(sync_cpu_, "1", 1);
            (void)s;
        }

        // 5. Centroid extraction via connected components
        cv::Mat mask_fpga(HEIGHT, WIDTH, CV_8UC1, shared_mem_ + FRAME_BYTES);
        cv::Mat binary_mask = (mask_fpga > 0);
        cv::Mat labels, stats, centroids;
        int n_labels = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8);

        int count = 0;
        for (int i = 1; i < n_labels && count < max_features; i++) {
            float cx = static_cast<float>(centroids.at<double>(i, 0));
            float cy = static_cast<float>(centroids.at<double>(i, 1));

            // Scale back to original frame coordinates if needed
            if (need_scale) {
                cx *= scale_x;
                cy *= scale_y;
            }
            pts.emplace_back(cx, cy);
            count++;
        }
    }

private:
    bool is_valid_;
    int udma_fd_, mem_fd_, sync_dev_, sync_cpu_;
    uint8_t* shared_mem_;
    volatile uint32_t* axi_;

    void init_hardware() {
        uint64_t phys_in = 0;
        FILE* f_addr = fopen("/sys/class/u-dma-buf/udmabuf0/phys_addr", "r");
        if (!f_addr || fscanf(f_addr, "%lx", &phys_in) != 1) {
            std::cerr << "[OpenVINS-FPGA] Failed to read udmabuf0 phys_addr.\n";
            if (f_addr) fclose(f_addr);
            return;
        }
        fclose(f_addr);

        // Sysfs cache parameters
        FILE* f_sz = fopen("/sys/class/u-dma-buf/udmabuf0/sync_size", "w");
        if (f_sz) { fprintf(f_sz, "%zu\n", TOTAL_BYTES); fclose(f_sz); }
        FILE* f_dir = fopen("/sys/class/u-dma-buf/udmabuf0/sync_direction", "w");
        if (f_dir) { fprintf(f_dir, "3\n"); fclose(f_dir); }
        FILE* f_off = fopen("/sys/class/u-dma-buf/udmabuf0/sync_offset", "w");
        if (f_off) { fprintf(f_off, "0\n"); fclose(f_off); }

        udma_fd_ = open("/dev/udmabuf0", O_RDWR);
        if (udma_fd_ < 0) return;
        shared_mem_ = (uint8_t*)mmap(NULL, TOTAL_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, udma_fd_, 0);
        if (shared_mem_ == MAP_FAILED) return;

        mem_fd_ = open("/dev/mem", O_RDWR | O_SYNC);
        if (mem_fd_ < 0) return;
        axi_ = (volatile uint32_t*)mmap(NULL, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd_, 0xB0000000);
        if (axi_ == MAP_FAILED) return;

        // Static AXI parameters
        uint64_t phys_out = phys_in + FRAME_BYTES;
        axi_[0x10 / 4] = (uint32_t)(phys_in & 0xFFFFFFFF);
        axi_[0x14 / 4] = (uint32_t)((phys_in >> 32) & 0xFFFFFFFF);
        axi_[0x1C / 4] = (uint32_t)(phys_out & 0xFFFFFFFF);
        axi_[0x20 / 4] = (uint32_t)((phys_out >> 32) & 0xFFFFFFFF);
        axi_[0x28 / 4] = HEIGHT;
        axi_[0x30 / 4] = WIDTH;
        axi_[0x38 / 4] = THRESHOLD;
        axi_[0x40 / 4] = K_SCALED;

        sync_dev_ = open("/sys/class/u-dma-buf/udmabuf0/sync_for_device", O_WRONLY);
        sync_cpu_ = open("/sys/class/u-dma-buf/udmabuf0/sync_for_cpu", O_WRONLY);

        is_valid_ = true;
        std::cout << "[OpenVINS-FPGA] Hardware Accelerator Initialized at 0xB0000000.\n";
    }
};

#endif // HCD_ACCELERATOR_HPP