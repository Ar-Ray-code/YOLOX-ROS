#pragma once

#include <cuda_runtime.h>
#include <opencv2/core/types.hpp>

__global__ void gpuResizeAndBlobFromImage(uchar3* image_data, float* output, int r_width, int r_height, 
                                            int input_width, int input_height, float w_ratio, float h_ratio);
extern "C" void launchGPUResizeAndBlobFromImage(uchar3* image_data, float* output, int r_width, int r_height, 
                                    int input_width, int input_height, float w_ratio, float h_ratio, cudaStream_t stream);