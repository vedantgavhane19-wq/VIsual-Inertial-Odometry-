#include "xf_corner_harris_config.h"

void corner_harris_accel(
    axis_stream_t& src_stream,
    axis_stream_t& dst_stream,
    int rows,
    int cols,
    int threshold,
    int k_scaled
) {
    // AXI-Lite Slave Interface for PS Control
    #pragma HLS INTERFACE s_axilite port=rows       bundle=control
    #pragma HLS INTERFACE s_axilite port=cols       bundle=control
    #pragma HLS INTERFACE s_axilite port=threshold  bundle=control
    #pragma HLS INTERFACE s_axilite port=k_scaled   bundle=control
    #pragma HLS INTERFACE s_axilite port=return     bundle=control

    // AXI4-Stream High-Speed Video Interfaces
    #pragma HLS INTERFACE axis port=src_stream register
    #pragma HLS INTERFACE axis port=dst_stream register

    // Concurrent Pipeline Dataflow
    #pragma HLS DATAFLOW

    // Internal Streaming Mat Objects
    xf::cv::Mat<TYPE, MAX_HEIGHT, MAX_WIDTH, NPC> in_mat(rows, cols);
    xf::cv::Mat<TYPE, MAX_HEIGHT, MAX_WIDTH, NPC> out_mat(rows, cols);
    #pragma HLS STREAM variable=in_mat.data depth=2
    #pragma HLS STREAM variable=out_mat.data depth=2

    // 1. Ingest AXI-Stream into xf::cv::Mat
    xf::cv::AXIvideo2xfMat(src_stream, in_mat);

    // 2. Hardware Harris Corner Pipeline
    xf::cv::cornerHarris<FILTER_WIDTH, BLOCK_WIDTH, NMS_RADIUS, TYPE, MAX_HEIGHT, MAX_WIDTH, NPC, USE_URAM>(
        in_mat, 
        out_mat, 
        threshold, 
        k_scaled
    );

    // 3. Egress xf::cv::Mat back to AXI-Stream
    xf::cv::xfMat2AXIvideo(out_mat, dst_stream);
}