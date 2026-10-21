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

   // --- 4. WEBCAM CAPTURE ---
    printf("4. Starting Webcam and warming up auto-exposure...\n");
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    cv::Mat frame, gray, img_fpga;
    
    // NEW: Read and discard 30 frames to let the camera brighten up
    for(int i = 0; i < 30; i++) {
        cap >> frame;
    }
    
    if (frame.empty()) {
        printf("Error: Webcam returned an empty 0-byte frame.\n");
        return -1;
    }

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, img_fpga, cv::Size(width, height));
    
    // NEW: Save the input image so you can physically see what the FPGA sees
    cv::imwrite("debug_input.jpg", img_fpga);

    // --- 5. EXECUTION ---
    printf("5. Formatting and copying to DMA...\n");
    memcpy(shared_mem, img_fpga.data, frame_bytes);
    memset(shared_mem + frame_bytes, 0, frame_bytes); 

    int sync_dev = open("/sys/class/u-dma-buf/udmabuf0/sync_for_device", O_WRONLY);
    if(sync_dev >= 0) { 
        ssize_t dummy = write(sync_dev, "1", 1); 
        (void)dummy; 
        close(sync_dev); 
    }

    printf("6. Triggering FPGA...\n");
    
    // NEW: Lower the threshold from 20 to 5 to force detections
    axi[0x38/4] = 5; 
    
    axi[0x00/4] = 0x1;

    int timeout = 0;
    while (!(axi[0x00/4] & 0x2)) {
        timeout++;
        if (timeout > 1000000) {
            printf("[FPGA ERROR] Timeout! Hardware deadlocked.\n");
            return -1;
        }
        usleep(10);
    }
    printf("7. FPGA Finished Successfully! Done signal received.\n");

    int sync_cpu = open("/sys/class/u-dma-buf/udmabuf0/sync_for_cpu", O_WRONLY);
    if(sync_cpu >= 0) { 
        ssize_t dummy = write(sync_cpu, "1", 1); 
        (void)dummy; 
        close(sync_cpu); 
    }

    cv::Mat mask_fpga(height, width, CV_8UC1, shared_mem + frame_bytes);
    int non_zero = cv::countNonZero(mask_fpga);
    printf("8. Test Complete! Corners detected: %d\n", non_zero);
    
    // NEW: Multiply by 255 to make the corners visible (white) and save it
    cv::Mat visible_mask = mask_fpga * 255;
    cv::imwrite("debug_corners.jpg", visible_mask);

    return 0;
}








#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <opencv2/opencv.hpp>

#define IP_BASE_ADDR 0xB0000000
#define IP_MAP_SIZE  0x10000

int main() {
    const int width = 480, height = 320;
    const size_t frame_bytes = width * height;
    const uint64_t phys_in  = 0x39200000ULL; // Your active physical address
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
    if (mem_fd < 0) { perror("Failed to open /dev/mem"); return 1; }
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

    std::cout << "4. Opening webcam..." << std::endl;
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open webcam" << std::endl;
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G')); // lighter on USB bandwidth/CPU

    // Mats wrapping the shared DDR buffers (no copies)
    cv::Mat gray(height, width, CV_8UC1, shared_mem);               // FPGA input
    cv::Mat mask(height, width, CV_8UC1, shared_mem + frame_bytes); // FPGA output

    // Open the cache-sync nodes once instead of every frame
    int sync_dev = open("/sys/class/u-dma-buf/udmabuf0/sync_for_device", O_WRONLY);
    int sync_cpu = open("/sys/class/u-dma-buf/udmabuf0/sync_for_cpu",    O_WRONLY);

    // If there is no display (headless board), save frames instead of using imshow
    const bool show = (getenv("DISPLAY") != nullptr);
    if (!show) std::cout << "No DISPLAY found: saving every 30th frame to out.png" << std::endl;

    cv::Mat frame, tmp, display, dilated;
    int frame_idx = 0;

    std::cout << "5. Running (press 'q' or ESC to quit, Ctrl+C if headless)..." << std::endl;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Failed to grab frame" << std::endl;
            break;
        }

        // Convert to grayscale at the exact resolution the FPGA expects
        cv::cvtColor(frame, tmp, cv::COLOR_BGR2GRAY);
        if (tmp.size() != gray.size())
            cv::resize(tmp, tmp, gray.size());
        tmp.copyTo(gray);                                   // writes into shared_mem

        memset(shared_mem + frame_bytes, 0, frame_bytes);   // clear output region

        // Flush CPU cache to DDR before the FPGA reads
        if (sync_dev >= 0) write(sync_dev, "1", 1);

        axi[0x00/4] = 0x1;
        while (!(axi[0x00/4] & 0x2)) {}

        // Invalidate CPU cache so we see what the FPGA wrote to DDR
        if (sync_cpu >= 0) write(sync_cpu, "1", 1);

        // Visualize: overlay detected corners in red on the grayscale image
        cv::cvtColor(gray, display, cv::COLOR_GRAY2BGR);
        cv::dilate(mask, dilated, cv::Mat(), cv::Point(-1, -1), 2); // make single pixels visible
        display.setTo(cv::Scalar(0, 0, 255), dilated);

        cv::putText(display, "Corners: " + std::to_string(cv::countNonZero(mask)),
                    cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2);

        if (show) {
            cv::imshow("FPGA Corner Detection", display);
            cv::imshow("Mask", mask);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;
        } else if (frame_idx % 30 == 0) {
            cv::imwrite("out.png", display);
        }
        frame_idx++;
    }

    std::cout << "6. Done." << std::endl;

    cap.release();
    cv::destroyAllWindows();
    if (sync_dev >= 0) close(sync_dev);
    if (sync_cpu >= 0) close(sync_cpu);

    munmap((void*)axi, IP_MAP_SIZE);
    munmap(shared_mem, frame_bytes * 2);
    close(mem_fd); close(udma_fd);
    return 0;
}
