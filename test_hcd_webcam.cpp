#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>
#include <chrono>
#include <csignal>
#include <opencv2/opencv.hpp>

// Flag for graceful exit on Ctrl+C
static volatile bool g_running = true;
void sigint_handler(int) {
    g_running = false;
}

// Configure u-dma-buf cache synchronization parameters in sysfs
bool setup_udmabuf_sync(size_t buffer_size) {
    FILE* f_sz = fopen("/sys/class/u-dma-buf/udmabuf0/sync_size", "w");
    if (!f_sz) return false;
    fprintf(f_sz, "%zu\n", buffer_size);
    fclose(f_sz);

    FILE* f_dir = fopen("/sys/class/u-dma-buf/udmabuf0/sync_direction", "w");
    if (!f_dir) return false;
    fprintf(f_dir, "3\n"); // 3 = DMA_BIDIRECTIONAL
    fclose(f_dir);

    FILE* f_off = fopen("/sys/class/u-dma-buf/udmabuf0/sync_offset", "w");
    if (!f_off) return false;
    fprintf(f_off, "0\n");
    fclose(f_off);

    return true;
}

int main() {
    signal(SIGINT, sigint_handler);

    const int WIDTH = 480;
    const int HEIGHT = 320;
    const size_t frame_bytes = WIDTH * HEIGHT;
    const size_t total_bytes = frame_bytes * 2;
    const int THRESHOLD = 1500;
    const int K_SCALED  = 2621;

    // 1. Read Physical DMA Address
    uint64_t phys_in = 0;
    FILE* f_addr = fopen("/sys/class/u-dma-buf/udmabuf0/phys_addr", "r");
    if (!f_addr || fscanf(f_addr, "%lx", &phys_in) != 1) {
        std::cerr << "Error: Cannot read udmabuf0 phys_addr. Is bitstream loaded?\n";
        if (f_addr) fclose(f_addr);
        return -1;
    }
    fclose(f_addr);

    setup_udmabuf_sync(total_bytes);

    // 2. Map udmabuf0 for DMA buffers
    int udma_fd = open("/dev/udmabuf0", O_RDWR);
    if (udma_fd < 0) {
        perror("Failed to open /dev/udmabuf0");
        return -1;
    }
    uint8_t* shared_mem = (uint8_t*)mmap(NULL, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, udma_fd, 0);
    if (shared_mem == MAP_FAILED) {
        perror("Failed to mmap udmabuf0");
        close(udma_fd);
        return -1;
    }
    uint8_t* in_buf  = shared_mem;
    uint8_t* out_buf = shared_mem + frame_bytes;

    // 3. Map AXI Slave Registers via /dev/mem
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        perror("Failed to open /dev/mem (run with sudo)");
        munmap(shared_mem, total_bytes);
        close(udma_fd);
        return -1;
    }
    volatile uint32_t* axi = (volatile uint32_t*)mmap(NULL, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0xB0000000);
    if (axi == MAP_FAILED) {
        perror("Failed to mmap AXI control registers");
        close(mem_fd);
        munmap(shared_mem, total_bytes);
        close(udma_fd);
        return -1;
    }

    // 4. Configure AXI Static Registers Once
    uint64_t phys_out = phys_in + frame_bytes;
    axi[0x10 / 4] = (uint32_t)(phys_in & 0xFFFFFFFF);
    axi[0x14 / 4] = (uint32_t)((phys_in >> 32) & 0xFFFFFFFF);
    axi[0x1C / 4] = (uint32_t)(phys_out & 0xFFFFFFFF);
    axi[0x20 / 4] = (uint32_t)((phys_out >> 32) & 0xFFFFFFFF);
    axi[0x28 / 4] = HEIGHT;
    axi[0x30 / 4] = WIDTH;
    axi[0x38 / 4] = THRESHOLD;
    axi[0x40 / 4] = K_SCALED;

    // Pre-open sync file descriptors to minimize syscall latency inside the loop
    int sync_dev = open("/sys/class/u-dma-buf/udmabuf0/sync_for_device", O_WRONLY);
    int sync_cpu = open("/sys/class/u-dma-buf/udmabuf0/sync_for_cpu", O_WRONLY);

    // 5. Initialize Webcam with V4L2 and minimal buffering for zero-lag streaming
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open webcam /dev/video0.\n";
        goto cleanup;
    }

    cv::namedWindow("FPGA Harris Corner Detection", cv::WINDOW_AUTOSIZE);

    {
        cv::Mat frame, gray;
        cv::Mat mask_fpga(HEIGHT, WIDTH, CV_8UC1, out_buf);
        cv::Mat labels, stats, centroids;

        int frame_count = 0;
        double current_fps = 0.0;
        auto fps_timer = std::chrono::steady_clock::now();

        std::cout << "\n>>> Streaming started. Press 'q' or 'ESC' in the window to stop. <<<\n\n";

        while (g_running) {
            cap >> frame;
            if (frame.empty()) continue;

            if (frame.cols != WIDTH || frame.rows != HEIGHT) {
                cv::resize(frame, frame, cv::Size(WIDTH, HEIGHT));
            }

            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            // Copy input frame to DMA input buffer
            memcpy(in_buf, gray.data, frame_bytes);

            // Flush CPU cache to DDR for FPGA
            if (sync_dev >= 0) {
                lseek(sync_dev, 0, SEEK_SET);
                ssize_t s = write(sync_dev, "1", 1);
                (void)s;
            }

            // --- TRIGGER FPGA & MEASURE LATENCY ---
            auto t_hw_start = std::chrono::high_resolution_clock::now();

            axi[0x00 / 4] = 0x01; // AP_START

            // Hardware busy-wait polling
            while (!(axi[0x00 / 4] & 0x02)) {
                // busy wait
            }

            auto t_hw_end = std::chrono::high_resolution_clock::now();
            double fpga_latency_ms = std::chrono::duration<double, std::milli>(t_hw_end - t_hw_start).count();

            // Clear control register
            axi[0x00 / 4] = 0x00;

            // Invalidate CPU cache to read FPGA results from DDR
            if (sync_cpu >= 0) {
                lseek(sync_cpu, 0, SEEK_SET);
                ssize_t s = write(sync_cpu, "1", 1);
                (void)s;
            }

            // --- EXTRACT CENTROIDS & DRAW OVERLAY ---
            cv::Mat binary_mask = (mask_fpga > 0);
            int n_labels = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8);
            int corners_count = n_labels - 1;

            // Draw detected corners (green dots)
            for (int i = 1; i < n_labels && i <= 1000; i++) {
                double cx = centroids.at<double>(i, 0);
                double cy = centroids.at<double>(i, 1);
                cv::circle(frame, cv::Point(cvRound(cx), cvRound(cy)), 3, cv::Scalar(0, 255, 0), -1);
            }

            // Calculate rolling FPS
            frame_count++;
            auto now = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - fps_timer).count();
            if (elapsed_sec >= 0.5) {
                current_fps = frame_count / elapsed_sec;
                frame_count = 0;
                fps_timer = now;
            }

            // HUD Overlay (FPS, FPGA time, corners count)
            char hud_buf[128];
            snprintf(hud_buf, sizeof(hud_buf), "FPS: %.1f | Corners: %d", current_fps, corners_count);
            cv::putText(frame, hud_buf, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 3);
            cv::putText(frame, hud_buf, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);

            snprintf(hud_buf, sizeof(hud_buf), "PL (FPGA): %.2f ms", fpga_latency_ms);
            cv::putText(frame, hud_buf, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 3);
            cv::putText(frame, hud_buf, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 1);

            cv::imshow("FPGA Harris Corner Detection", frame);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }
    }

cleanup:
    std::cout << "\nClosing devices and unmapping memory...\n";
    if (sync_dev >= 0) close(sync_dev);
    if (sync_cpu >= 0) close(sync_cpu);
    cap.release();
    cv::destroyAllWindows();
    munmap((void*)axi, 0x10000);
    close(mem_fd);
    munmap(shared_mem, total_bytes);
    close(udma_fd);
    std::cout << "Done.\n";
    return 0;
}
