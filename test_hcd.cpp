#include <iostream>
#include <cstdint>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <opencv2/opencv.hpp>

#define IP_BASE_ADDR 0xB0000000
#define IP_MAP_SIZE  0x10000

#define CTRL     (0x00 / 4)
#define SRC_LO   (0x10 / 4)
#define SRC_HI   (0x14 / 4)
#define DST_LO   (0x1C / 4)
#define DST_HI   (0x20 / 4)
#define ROWS     (0x28 / 4)
#define COLS     (0x30 / 4)
#define THRESH   (0x38 / 4)
#define K_SCALE  (0x40 / 4)

int main() {
    const int width = 480, height = 320;
    const size_t frame_bytes = width * height;
    const uint64_t phys_in  = 0x39000000ULL;
    const uint64_t phys_out = phys_in + frame_bytes;

    int udma_fd = open("/dev/udmabuf0", O_RDWR | O_SYNC);
    uint8_t* shared_mem = (uint8_t*)mmap(NULL, frame_bytes * 2, PROT_READ | PROT_WRITE, MAP_SHARED, udma_fd, 0);
    uint8_t* in_buf  = shared_mem;
    uint8_t* out_buf = shared_mem + frame_bytes;

    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    volatile uint32_t* axi = (volatile uint32_t*)mmap(NULL, IP_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, IP_BASE_ADDR);

    axi[SRC_LO]  = (uint32_t)(phys_in & 0xFFFFFFFF);
    axi[SRC_HI]  = (uint32_t)(phys_in >> 32);
    axi[DST_LO]  = (uint32_t)(phys_out & 0xFFFFFFFF);
    axi[DST_HI]  = (uint32_t)(phys_out >> 32);
    axi[ROWS]    = height;
    axi[COLS]    = width;
    axi[THRESH]  = 150;
    axi[K_SCALE] = 2621;

    cv::Mat gray(height, width, CV_8UC1, in_buf);
    gray.setTo(0);
    for (int y = 40; y < 280; y += 40) {
        for (int x = 40; x < 440; x += 40) {
            if (((x / 40) + (y / 40)) % 2 == 0)
                cv::rectangle(gray, cv::Rect(x, y, 40, 40), cv::Scalar(255), -1);
        }
    }
    memset(out_buf, 0, frame_bytes);

    auto t0 = std::chrono::high_resolution_clock::now();
    axi[CTRL] = 0x1; // AP_START
    while (!(axi[CTRL] & 0x2)) {} // Wait for AP_DONE
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    cv::Mat mask(height, width, CV_8UC1, out_buf);
    int nonzero = cv::countNonZero(mask);

    std::cout << "FPGA Execution Success! Latency: " << ms << " ms | Corner Pixels: " << nonzero << std::endl;

    munmap((void*)axi, IP_MAP_SIZE);
    munmap(shared_mem, frame_bytes * 2);
    close(mem_fd); close(udma_fd);
    return 0;
}