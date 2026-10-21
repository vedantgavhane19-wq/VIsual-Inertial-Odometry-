#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>
#include <opencv2/opencv.hpp>

// Helper to configure u-dma-buf cache synchronization parameters
void configure_udmabuf_sync(size_t size) {
    FILE* f_sz = fopen("/sys/class/u-dma-buf/udmabuf0/sync_size", "w");
    if (f_sz) {
        fprintf(f_sz, "%zu\n", size);
        fclose(f_sz);
    }
    FILE* f_dir = fopen("/sys/class/u-dma-buf/udmabuf0/sync_direction", "w");
    if (f_dir) {
        fprintf(f_dir, "3\n"); // 3 = DMA_BIDIRECTIONAL
        fclose(f_dir);
    }
}

int main() {
    // 1. Read physical base address and buffer size from sysfs
    uint64_t phys_in = 0;
    FILE* f = fopen("/sys/class/u-dma-buf/udmabuf0/phys_addr", "r");
    if (f) {
        if (fscanf(f, "%lx", &phys_in) != 1) {
            std::cerr << "Error: Failed to parse physical address.\n";
            fclose(f);
            return -1;
        }
        fclose(f);
    } else {
        std::cerr << "Error: Cannot open udmabuf0 phys_addr. Is the bitstream loaded?\n";
        return -1;
    }

    size_t actual_buf_size = 0;
    FILE* f_sz = fopen("/sys/class/u-dma-buf/udmabuf0/size", "r");
    if (f_sz) {
        if (fscanf(f_sz, "%zu", &actual_buf_size) == 1) {
            std::cout << "[0] udmabuf0 Size: " << actual_buf_size << " bytes\n";
        }
        fclose(f_sz);
    }
    std::cout << "[0] FPGA Physical DMA Address: 0x" << std::hex << phys_in << std::dec << "\n";

    const int width = 480;
    const int height = 320;
    const size_t frame_bytes = width * height;
    const size_t total_bytes = frame_bytes * 2;
    const int threshold = 5;
    const int k_param = 2621;

    if (actual_buf_size > 0 && actual_buf_size < total_bytes) {
        std::cerr << "Error: udmabuf0 (" << actual_buf_size << " bytes) is too small! Need: " << total_bytes << " bytes.\n";
        return -1;
    }

    // Configure sync parameters for u-dma-buf
    configure_udmabuf_sync(total_bytes);

    // 2. Open udmabuf0 as normal cacheable memory (NO O_SYNC to avoid NEON Bus Error)
    std::cout << "[1] Opening udmabuf0 (Cacheable)...\n";
    int udma_fd = open("/dev/udmabuf0", O_RDWR);
    if (udma_fd < 0) {
        perror("Error opening /dev/udmabuf0");
        return -1;
    }
    uint8_t* shared_mem = (uint8_t*)mmap(NULL, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, udma_fd, 0);
    if (shared_mem == MAP_FAILED) {
        perror("Error mmap udmabuf0");
        close(udma_fd);
        return -1;
    }

    // 3. Open AXI registers via /dev/mem (O_SYNC is required for MMIO)
    std::cout << "[2] Opening AXI Registers via /dev/mem...\n";
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        perror("Error opening /dev/mem (must run with sudo)");
        munmap(shared_mem, total_bytes);
        close(udma_fd);
        return -1;
    }
    volatile uint32_t* axi = (volatile uint32_t*)mmap(NULL, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0xB0000000);
    if (axi == MAP_FAILED) {
        perror("Error mmap AXI registers");
        close(mem_fd);
        munmap(shared_mem, total_bytes);
        close(udma_fd);
        return -1;
    }

    // 4. Configure AXI Core Registers
    std::cout << "[3] Writing AXI configuration...\n";
    axi[0x10 / 4] = (uint32_t)(phys_in & 0xFFFFFFFF);
    axi[0x14 / 4] = (uint32_t)(phys_in >> 32);
    axi[0x1C / 4] = (uint32_t)((phys_in + frame_bytes) & 0xFFFFFFFF);
    axi[0x20 / 4] = (uint32_t)((phys_in + frame_bytes) >> 32);
    axi[0x28 / 4] = height;
    axi[0x30 / 4] = width;
    axi[0x38 / 4] = threshold;
    axi[0x40 / 4] = k_param;

    // 5. Capture Webcam Frame
    std::cout << "[4] Capturing frame from webcam...\n";
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open /dev/video0.\n";
        munmap((void*)axi, 0x10000);
        close(mem_fd);
        munmap(shared_mem, total_bytes);
        close(udma_fd);
        return -1;
    }

    cv::Mat frame, gray, img_fpga;
    for (int i = 0; i < 30; i++) {
        cap >> frame;
    }

    if (frame.empty()) {
        std::cerr << "Error: Captured frame is empty.\n";
        munmap((void*)axi, 0x10000);
        close(mem_fd);
        munmap(shared_mem, total_bytes);
        close(udma_fd);
        return -1;
    }

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, img_fpga, cv::Size(width, height));
    cv::imwrite("debug_input.jpg", img_fpga);

    // 6. Copy input image & clear output space in DMA memory
    std::cout << "[5] Copying image data to physical DMA buffer...\n";
    memcpy(shared_mem, img_fpga.data, frame_bytes);
    memset(shared_mem + frame_bytes, 0, frame_bytes);

    // Flush CPU L1/L2 caches to DDR memory for the FPGA
    int sync_dev = open("/sys/class/u-dma-buf/udmabuf0/sync_for_device", O_WRONLY);
    if (sync_dev >= 0) {
        ssize_t s = write(sync_dev, "1", 1);
        (void)s;
        close(sync_dev);
    }

    // 7. AXI Handshake & Polling
    std::cout << "[6] Triggering FPGA execution...\n";
    axi[0x00 / 4] = 0x00;
    usleep(50);
    axi[0x00 / 4] = 0x01; // Start IP

    int timeout = 0;
    while (!(axi[0x00 / 4] & 0x02)) {
        timeout++;
        if (timeout > 2000000) {
            std::cerr << "\n[FPGA ERROR] Timeout! Hardware deadlocked. Status: 0x"
                      << std::hex << axi[0x00 / 4] << std::dec << "\n";
            munmap((void*)axi, 0x10000);
            close(mem_fd);
            munmap(shared_mem, total_bytes);
            close(udma_fd);
            return -1;
        }
        usleep(10);
    }
    std::cout << "[7] FPGA completed! Polling cycles: " << timeout
              << " | Final Status: 0x" << std::hex << axi[0x00 / 4] << std::dec << "\n";

    axi[0x00 / 4] = 0x00;

    // Invalidate CPU cache so it reads fresh DDR memory written by the FPGA
    int sync_cpu = open("/sys/class/u-dma-buf/udmabuf0/sync_for_cpu", O_WRONLY);
    if (sync_cpu >= 0) {
        ssize_t s = write(sync_cpu, "1", 1);
        (void)s;
        close(sync_cpu);
    }

    // 8. Output Analysis & Visualization
    cv::Mat mask_fpga(height, width, CV_8UC1, shared_mem + frame_bytes);
    int corners_found = cv::countNonZero(mask_fpga);

    double min_val = 0, max_val = 0;
    cv::minMaxLoc(mask_fpga, &min_val, &max_val);

    std::cout << "[8] Detection Results:\n"
              << "    - Total corners: " << corners_found << "\n"
              << "    - Min pixel value: " << min_val << "\n"
              << "    - Max pixel value: " << max_val << "\n";

    cv::Mat visible_mask;
    if (max_val == 1) {
        visible_mask = mask_fpga * 255;
    } else {
        visible_mask = mask_fpga.clone();
    }
    cv::imwrite("debug_corners.jpg", visible_mask);

    cv::Mat overlay;
    cv::resize(frame, overlay, cv::Size(width, height));
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            if (mask_fpga.at<uint8_t>(r, c) > 0) {
                cv::circle(overlay, cv::Point(c, r), 3, cv::Scalar(0, 0, 255), -1);
            }
        }
    }
    cv::imwrite("debug_overlay.jpg", overlay);
    std::cout << "    - Saved 'debug_corners.jpg' and 'debug_overlay.jpg'\n";

    // Clean up
    munmap((void*)axi, 0x10000);
    close(mem_fd);
    munmap(shared_mem, total_bytes);
    close(udma_fd);
    return 0;
}
