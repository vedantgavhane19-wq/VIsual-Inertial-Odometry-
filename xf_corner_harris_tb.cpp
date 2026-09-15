#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include "xf_corner_harris_config.h"

// Saves raw grayscale buffers into viewable .pgm images
void save_pgm(const char* filename, const std::vector<uint8_t>& img, int rows, int cols) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "[ERROR] Could not open file: " << filename << std::endl;
        return;
    }
    ofs << "P5\n" << cols << " " << rows << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(img.data()), rows * cols);
    ofs.close();
}

int main() {
    const int rows = MAX_HEIGHT;
    const int cols = MAX_WIDTH;
    const int threshold = 442;
    const int k_scaled = 2621; // 0.04 * 65536

    std::cout << "[TB INFO] Generating 1920x1080 synthetic test frame..." << std::endl;
    std::vector<uint8_t> input_frame(rows * cols, 0);

    // Draw high-contrast boxes to generate corner points
    for (int r = 200; r < 600; r++) {
        for (int c = 200; c < 600; c++) {
            input_frame[r * cols + c] = 255;
        }
    }
    for (int r = 300; r < 800; r++) {
        for (int c = 800; c < 1400; c++) {
            input_frame[r * cols + c] = 200;
        }
    }

    axis_stream_t src_stream;
    axis_stream_t dst_stream;

    // Pack frame into AXI4-Stream
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            axis_pixel_t pix;
            pix.data = input_frame[r * cols + c];
            pix.keep = -1;
            pix.strb = -1;
            pix.user = (r == 0 && c == 0) ? 1 : 0;
            pix.last = (c == cols - 1) ? 1 : 0;
            src_stream.write(pix);
        }
    }

    std::cout << "[TB INFO] Invoking corner_harris_accel kernel..." << std::endl;
    corner_harris_accel(src_stream, dst_stream, rows, cols, threshold, k_scaled);

    // Read back output stream and tally detected corners
    std::vector<uint8_t> output_frame(rows * cols, 0);
    int corner_count = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            axis_pixel_t out_pix = dst_stream.read();
            output_frame[r * cols + c] = out_pix.data;
            if (out_pix.data == 255) {
                corner_count++;
            }
        }
    }

    std::cout << "[TB SUCCESS] Simulation complete! Detected corners: " << corner_count << std::endl;
    save_pgm("input_test_pattern.pgm", input_frame, rows, cols);
    save_pgm("corners_detected.pgm", output_frame, rows, cols);

    return (corner_count > 0) ? 0 : 1;
}