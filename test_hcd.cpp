#include <iostream>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <opencv2/opencv.hpp>

#define IP_BASE_ADDR 0xB0000000 // We will verify this if it crashes here
#define IP_MAP_SIZE  0x10000

int main() {
    const int width = 480, height = 320;
    const size_t frame_bytes = width * height;
    const uint64_t phys_in  = 0x39000000ULL;
    const uint64_t phys_out = phys_in + frame_bytes;

    std::cout << "1. Mapping udmabuf..." << std::endl;
    int udma_fd = open("/dev/udmabuf0", O_RDWR | O_SYNC);
    uint8_t* shared_mem = (uint8_t*)mmap(NULL, frame_bytes * 2, PROT_READ | PROT_WRITE, MAP_SHARED, udma_fd, 0);
    if (shared_mem == MAP_FAILED) { std::cout << "ERROR: udmabuf mmap failed!\n"; return 1; }

    std::cout << "2. Mapping /dev/mem..." << std::endl;
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    volatile uint32_t* axi = (volatile uint32_t*)mmap(NULL, IP_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, IP_BASE_ADDR);
    if (axi == MAP_FAILED) { std::cout << "ERROR: /dev/mem mmap failed!\n"; return 1; }

    std::cout << "3. Writing to AXI registers at 0x" << std::hex << IP_BASE_ADDR << std::dec << "..." << std::endl;
    // If 0xB0000000 is the wrong address in Vivado, the Segfault will happen exactly here:
    axi[0x10/4] = (uint32_t)(phys_in & 0xFFFFFFFF);
    axi[0x14/4] = (uint32_t)(phys_in >> 32);
    axi[0x28/4] = height;
    axi[0x30/4] = width;
    
    std::cout << "4. Generating test image..." << std::endl;
    cv::Mat gray(height, width, CV_8UC1, shared_mem);
    gray.setTo(255); // Fill with white
    
    std::cout << "5. Triggering AP_START..." << std::endl;
    axi[0x00/4] = 0x1; 
    
    std::cout << "6. Waiting for AP_DONE..." << std::endl;
    while (!(axi[0x00/4] & 0x2)) {} 
    
    std::cout << "7. Hardware Execution Complete!" << std::endl;
    
    munmap((void*)axi, IP_MAP_SIZE);
    munmap(shared_mem, frame_bytes * 2);
    close(mem_fd); close(udma_fd);
    return 0;
}
