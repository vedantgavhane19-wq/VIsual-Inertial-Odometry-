#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>
#include <opencv2/opencv.hpp>

int main() {
    // --- 1. CONFIGURATION ---
    // Note: Added the trailing '0' to your address. If your terminal explicitly 
    // printed only 7 digits, remove the last 0, but this is extremely rare.
    const uint64_t phys_in = 0x3a800000ULL; 
    
    const int width = 480;
    const int height = 320;
    const size_t frame_bytes = width * height;
    const int threshold = 20;

    // --- 2. MEMORY MAPPING ---
    printf("1. Opening udmabuf0...\n");
    int udma_fd = open("/dev/udmabuf0", O_RDWR);
    if (udma_fd < 0) { printf("Error: Cannot open /dev/udmabuf0\n"); return -1; }
    uint8_t* shared_mem = (uint8_t*)mmap(NULL, frame_bytes * 2, PROT_READ | PROT_WRITE, MAP_SHARED, udma_fd, 0);

    printf("2. Opening AXI Registers...\n");
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) { printf("Error: Cannot open /dev/mem\n"); return -1; }
    volatile uint32_t* axi = (volatile uint32_t*)mmap(NULL, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0xB0000000);

    if (shared_mem == MAP_FAILED || axi == MAP_FAILED) {
        printf("Error: Memory mapping failed!\n");
        return -1;
    }

    // --- 3. AXI INITIALIZATION ---
    printf("3. Writing AXI Configuration...\n");
    axi[0x10/4] = (uint32_t)(phys_in & 0xFFFFFFFF);
    axi[0x14/4] = (uint32_t)(phys_in >> 32);
    axi[0x1C/4] = (uint32_t)((phys_in + frame_bytes) & 0xFFFFFFFF);
    axi[0x20/4] = (uint32_t)((phys_in + frame_bytes) >> 32);
    axi[0x28/4] = height; // Rows
    axi[0x30/4] = width;  // Cols
    axi[0x38/4] = threshold;
    axi[0x40/4] = 2621;

    // --- 4. WEBCAM CAPTURE ---
    printf("4. Starting Webcam...\n");
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("Error: Cannot open webcam /dev/video0\n");
        return -1;
    }

    cv::Mat frame, gray, img_fpga;
    cap >> frame; // Grab a test frame
    
    if (frame.empty()) {
        printf("Error: Webcam returned an empty 0-byte frame.\n");
        return -1;
    }

    // Force strict formatting so the FPGA doesn't read out of bounds
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, img_fpga, cv::Size(width, height));

    // --- 5. EXECUTION ---
    printf("5. Formatting and copying to DMA...\n");
    memcpy(shared_mem, img_fpga.data, frame_bytes);
    memset(shared_mem + frame_bytes, 0, frame_bytes); // Wipe the output buffer

    int sync_dev = open("/sys/class/u-dma-buf/udmabuf0/sync_for_device", O_WRONLY);
    if(sync_dev >= 0) { if(write(sync_dev, "1", 1)<0){} close(sync_dev); }

    printf("6. Triggering FPGA...\n");
    
    // Hard reset the Vitis HLS state machine to clear any previous lockups
    axi[0x00/4] = 0x0; 
    usleep(1000); // 1ms wait
    
    // Trigger Start
    axi[0x00/4] = 0x1;

    // Polling Loop with Timeout
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
    if(sync_cpu >= 0) { if(write(sync_cpu, "1", 1)<0){} close(sync_cpu); }

    // Check how many pixels the FPGA marked as corners
    cv::Mat mask_fpga(height, width, CV_8UC1, shared_mem + frame_bytes);
    int non_zero = cv::countNonZero(mask_fpga);
    printf("8. Test Complete! Corners detected: %d\n", non_zero);

    return 0;
}
