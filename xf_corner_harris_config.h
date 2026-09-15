#ifndef _XF_CORNER_HARRIS_CONFIG_H_
#define _XF_CORNER_HARRIS_CONFIG_H_

#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "common/xf_infra.hpp"      // <--- REQUIRED for AXIvideo2xfMat and xfMat2AXIvideo
#include "features/xf_harris.hpp"

// Maximum Resolution (1080p stream for KV260)
#define MAX_HEIGHT 1080
#define MAX_WIDTH  1920

// Algorithm Parameters
#define FILTER_WIDTH 3    // Sobel gradient aperture (3x3)
#define BLOCK_WIDTH  3    // Structure tensor summation window (3x3)
#define NMS_RADIUS   1    // Non-Max Suppression radius (1 -> 3x3 search)

// Hardware Types
#define TYPE          XF_8UC1     // 8-bit Grayscale Single Channel
#define NPC           XF_NPPC1    // 1 Pixel Processed Per Clock (II=1)
#define USE_URAM      0           // 0: Use BRAM (Line buffers are small for 1080p)

// AXI Video Interface Types (8-bit data for grayscale)
typedef ap_axiu<8, 1, 1, 1> axis_pixel_t;
typedef hls::stream<axis_pixel_t> axis_stream_t;

// Top-level synthesis prototype
void corner_harris_accel(
    axis_stream_t& src_stream,
    axis_stream_t& dst_stream,
    int rows,
    int cols,
    int threshold,
    int k_scaled
);

#endif