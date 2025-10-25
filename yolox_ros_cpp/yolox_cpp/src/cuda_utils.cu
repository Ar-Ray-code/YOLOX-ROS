#include <cstdio>
#include <yolox_cpp/cuda_utils.cuh>

__global__ void gpuResizeAndBlobFromImage(uchar3* image_data, float* output, int r_width, int r_height, 
                                            int input_width, int input_height, float w_ratio, float h_ratio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= r_width || y >= r_height) return;

    // Look into the input image, grab all the pixels
    // Do math to determine the color of the output pixel; for bilinear scaling
    float sy = (y + 0.5) * h_ratio - 0.5;
    float sx = (x + 0.5) * w_ratio - 0.5;

    int y0 = (int) sy;
    int x0 = (int) sx;
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    y0 = max(y0, 0);
    x0 = max(x0, 0);
    y1 = min(y1, input_height - 1);
    x1 = min(x1, input_width - 1);

    int src_idx = y0 * input_width + x0;
    if (src_idx >= input_width * input_height) return;

    // Weights for each input pixel to contribute
    float w00 = (1 - (sy - y0)) * (1 - (sx - x0)); // For y = 0 x = 0
    float w01 = (1 - (sy - y0)) * (sx - x0); // For y = 0 x = 1
    float w10 = (sy - y0) * (1 - (sx - x0)); // For y = 1 x = 0 dy(1-dx)
    float w11 = (sy - y0) * (sx - x0); // For y = 1 x = 1

    int r_idx = 0 * r_height * r_width + y * r_width + x;
    int g_idx = 1 * r_height * r_width + y * r_width + x;
    int b_idx = 2 * r_height * r_width + y * r_width + x;
    
    uchar3 p00 = image_data[y0 * input_width + x0];
    uchar3 p01 = image_data[y0 * input_width + x1];
    uchar3 p10 = image_data[y1 * input_width + x0];
    uchar3 p11 = image_data[y1 * input_width + x1];

    // Red channel (uchar3.z)
    output[r_idx] = w00 * p00.z + w01 * p01.z + w10 * p10.z + w11 * p11.z;
    // Green channel (uchar3.y)
    output[g_idx] = w00 * p00.y + w01 * p01.y + w10 * p10.y + w11 * p11.y;
    // Blue channel (uchar3.x)
    output[b_idx] = w00 * p00.x + w01 * p01.x + w10 * p10.x + w11 * p11.x;
}

extern "C" void launchGPUResizeAndBlobFromImage(uchar3* image_data, float* output, int r_width, int r_height, 
                                    int input_width, int input_height, float w_ratio, float h_ratio, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((r_width + block.x - 1) / block.x, 
             (r_height + block.y - 1) / block.y);
    gpuResizeAndBlobFromImage<<<grid, block, 0, stream>>>(image_data, output, r_width, r_height, input_width, input_height, w_ratio, h_ratio);
}