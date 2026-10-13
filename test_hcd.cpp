#include <iostream>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <opencv2/opencv.hpp>

#define IP_BASE_ADDR 0xB0000000
#define IP_MAP_SIZE  0x10000

int main() {
    const int width = 480, height = 320;
    const size_t frame_bytes = width * height;
    const uint64_t phys_in  = 0x3b100000ULL; // Your active physical address
    const uint64_t phys_out = phys_in + frame_bytes;

    std::cout << "1. Mapping udmabuf (Cached)..." << std::endl;
    // Removed O_SYNC here to prevent OpenCV ARM NEON Bus Errors
    int udma_fd = open("/dev/udmabuf0", O_RDWR); 
    if (udma_fd < 0) { perror("Failed to open udmabuf"); return 1; }

    uint8_t* shared_mem = (uint8_t*)mmap(NULL, frame_bytes * 2, PROT_READ | PROT_WRITE, MAP_SHARED, udma_fd, 0);
    if (shared_mem == MAP_FAILED) { perror("Failed to mmap udmabuf"); return 1; }

    std::cout << "2. Mapping /dev/mem (Uncached)..." << std::endl;
    // O_SYNC must remain here because AXI registers are physical hardware switches
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC); 
    volatile uint32_t* axi = (volatile uint32_t*)mmap(NULL, IP_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, IP_BASE_ADDR);
    if (axi == MAP_FAILED) { perror("Failed to mmap /dev/mem"); return 1; }

    std::cout << "3. Configuring AXI registers..." << std::endl;
    axi[0x10/4] = (uint32_t)(phys_in & 0xFFFFFFFF);
    axi[0x14/4] = (uint32_t)(phys_in >> 32);
    axi[0x1C/4] = (uint32_t)(phys_out & 0xFFFFFFFF);
    axi[0x20/4] = (uint32_t)(phys_out >> 32);
    axi[0x28/4] = height;
    axi[0x30/4] = width;
    axi[0x38/4] = 150;
    axi[0x40/4] = 2621;

    std::cout << "4. Generating test image (OpenCV)..." << std::endl;
    cv::Mat gray(height, width, CV_8UC1, shared_mem);
    gray.setTo(0);
    for (int y = 40; y < 280; y += 40) {
        for (int x = 40; x < 440; x += 40) {
            if (((x / 40) + (y / 40)) % 2 == 0)
                cv::rectangle(gray, cv::Rect(x, y, 40, 40), cv::Scalar(255), -1);
        }
    }
    memset(shared_mem + frame_bytes, 0, frame_bytes);

    std::cout << "5. Syncing Cache & Triggering Hardware..." << std::endl;
    
    // Flush CPU cache to DDR before FPGA reads
    int sync_dev = open("/sys/class/u-dma-buf/udmabuf0/sync_for_device", O_WRONLY);
    if(sync_dev >= 0) { write(sync_dev, "1", 1); close(sync_dev); }

    axi[0x00/4] = 0x1;
    while (!(axi[0x00/4] & 0x2)) {}

    // Invalidate CPU cache so OpenCV sees what the FPGA wrote to DDR
    int sync_cpu = open("/sys/class/u-dma-buf/udmabuf0/sync_for_cpu", O_WRONLY);
    if(sync_cpu >= 0) { write(sync_cpu, "1", 1); close(sync_cpu); }

    std::cout << "6. Hardware Execution Complete!" << std::endl;
    cv::Mat mask(height, width, CV_8UC1, shared_mem + frame_bytes);
    std::cout << "Non-zero corner pixels detected: " << cv::countNonZero(mask) << std::endl;

    munmap((void*)axi, IP_MAP_SIZE);
    munmap(shared_mem, frame_bytes * 2);
    close(mem_fd); close(udma_fd);
    return 0;
}
