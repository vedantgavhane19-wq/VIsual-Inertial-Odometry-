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
    const uint64_t phys_in  = 0x39000000ULL;
    const uint64_t phys_out = phys_in + frame_bytes;

    std::cout << "1. Opening and mapping udmabuf..." << std::endl;
    int udma_fd = open("/dev/udmabuf0", O_RDWR | O_SYNC);
    if (udma_fd < 0) { 
        perror("Failed to open /dev/udmabuf0"); 
        return 1; 
    }

    uint8_t* shared_mem = (uint8_t*)mmap(NULL, frame_bytes * 2, PROT_READ | PROT_WRITE, MAP_SHARED, udma_fd, 0);
    if (shared_mem == MAP_FAILED) { 
        perror("Failed to mmap /dev/udmabuf0"); 
        close(udma_fd);
        return 1; 
    }

    std::cout << "2. Mapping /dev/mem..." << std::endl;
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) { 
        perror("Failed to open /dev/mem (run with sudo)"); 
        munmap(shared_mem, frame_bytes * 2);
        close(udma_fd);
        return 1; 
    }

    volatile uint32_t* axi = (volatile uint32_t*)mmap(NULL, IP_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, IP_BASE_ADDR);
    if (axi == MAP_FAILED) { 
        perror("Failed to mmap /dev/mem"); 
        munmap(shared_mem, frame_bytes * 2);
        close(mem_fd); close(udma_fd);
        return 1; 
    }

    std::cout << "3. Configuring AXI registers..." << std::endl;
    axi[0x10/4] = (uint32_t)(phys_in & 0xFFFFFFFF);
    axi[0x14/4] = (uint32_t)(phys_in >> 32);
    axi[0x1C/4] = (uint32_t)(phys_out & 0xFFFFFFFF);
    axi[0x20/4] = (uint32_t)(phys_out >> 32);
    axi[0x28/4] = height;
    axi[0x30/4] = width;
    axi[0x38/4] = 150;
    axi[0x40/4] = 2621;

    std::cout << "4. Generating test image..." << std::endl;
    cv::Mat gray(height, width, CV_8UC1, shared_mem);
    gray.setTo(0);
    for (int y = 40; y < 280; y += 40) {
        for (int x = 40; x < 440; x += 40) {
            if (((x / 40) + (y / 40)) % 2 == 0)
                cv::rectangle(gray, cv::Rect(x, y, 40, 40), cv::Scalar(255), -1);
        }
    }
    memset(shared_mem + frame_bytes, 0, frame_bytes);

    std::cout << "5. Triggering AP_START..." << std::endl;
    axi[0x00/4] = 0x1;

    std::cout << "6. Waiting for AP_DONE..." << std::endl;
    while (!(axi[0x00/4] & 0x2)) {}

    std::cout << "7. Hardware Execution Complete!" << std::endl;
    cv::Mat mask(height, width, CV_8UC1, shared_mem + frame_bytes);
    std::cout << "Non-zero corner pixels detected: " << cv::countNonZero(mask) << std::endl;

    munmap((void*)axi, IP_MAP_SIZE);
    munmap(shared_mem, frame_bytes * 2);
    close(mem_fd);
    close(udma_fd);
    return 0;
}
