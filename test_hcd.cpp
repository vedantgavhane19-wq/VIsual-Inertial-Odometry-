#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>
#include <opencv2/opencv.hpp>

int main() {
    // 1. Dynamically read the physical address to prevent bad memory targeting
    uint64_t phys_in = 0;
    FILE* f = fopen("/sys/class/u-dma-buf/udmabuf0/phys_addr", "r");
    if (f) {
        if (fscanf(f, "%lx", &phys_in) != 1) {
            printf("Error: Failed to parse physical address.\n");
            fclose(f);
            return -1;
        }
        fclose(f);
    } else {
        printf("Error: Cannot open /sys/class/u-dma-buf/udmabuf0/phys_addr. Is the bitstream loaded?\n");
        return -1;
    }
    printf("0. Using Dynamic FPGA Address: 0x%lX\n", phys_in);

    const int width = 480;
    const int height = 320;
    const size_t frame_bytes = width * height;
    const int threshold = 20;

    // 2. Memory Mapping
    printf("1. Opening udmabuf0...\n");
    int udma_fd = open("/dev/udmabuf0", O_RDWR);
    if (udma_fd < 0) return -1;
    uint8_t* shared_mem = (uint8_t*)mmap(NULL, frame_bytes * 2, PROT_READ | PROT_WRITE, MAP_SHARED, udma_fd, 0);

    printf("2. Opening AXI Registers...\n");
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) return -1;
    volatile uint32_t* axi = (volatile uint32_t*)mmap(NULL, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0xB0000000);

    // 3. AXI Configuration
    printf("3. Writing AXI Configuration...\n");
    axi[0x10/4] = (uint32_t)(phys_in & 0xFFFFFFFF);
    axi[0x14/4] = (uint32_t)(phys_in >> 32);
    axi[0x1C/4] = (uint32_t)((phys_in + frame_bytes) & 0xFFFFFFFF);
    axi[0x20/4] = (uint32_t)((phys_in + frame_bytes) >> 32);
    axi[0x28/4] = height; 
    axi[0x30/4] = width;  
    axi[0x38/4] = threshold;
    axi[0x40/4] = 2621;

    // 4. Webcam Capture
    printf("4. Starting Webcam...\n");
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    cv::Mat frame, gray, img_fpga;
    cap >> frame; 
    if (frame.empty()) {
        printf("Error: Webcam returned an empty 0-byte frame.\n");
        return -1;
    }

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, img_fpga, cv::Size(width, height));

    // 5. Execution
    printf("5. Formatting and copying to DMA...\n");
    memcpy(shared_mem, img_fpga.data, frame_bytes);
    memset(shared_mem + frame_bytes, 0, frame_bytes); 

    int sync_dev = open("/sys/class/u-dma-buf/udmabuf0/sync_for_device", O_WRONLY);
    if(sync_dev >= 0) { 
        ssize_t dummy = write(sync_dev, "1", 1); 
        (void)dummy; // Suppress compiler warning
        close(sync_dev); 
    }

    printf("6. Triggering FPGA...\n");
    
    // Trigger Start directly (Removed the 0x0 reset that causes AXI freezes)
    axi[0x00/4] = 0x1;

    int timeout = 0;
    while (!(axi[0x00/4] & 0x2)) {
        timeout++;
        if (timeout > 1000000) {
            printf("[FPGA ERROR] Timeout! Hardware deadlocked. AXI Control State: 0x%X\n", axi[0x00/4]);
            return -1;
        }
        usleep(10);
    }
    printf("7. FPGA Finished Successfully! Done signal received.\n");

    int sync_cpu = open("/sys/class/u-dma-buf/udmabuf0/sync_for_cpu", O_WRONLY);
    if(sync_cpu >= 0) { 
        ssize_t dummy = write(sync_cpu, "1", 1); 
        (void)dummy; // Suppress compiler warning
        close(sync_cpu); 
    }

    cv::Mat mask_fpga(height, width, CV_8UC1, shared_mem + frame_bytes);
    int non_zero = cv::countNonZero(mask_fpga);
    printf("8. Test Complete! Corners detected: %d\n", non_zero);

    return 0;
}
