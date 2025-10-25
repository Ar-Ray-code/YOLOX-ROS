#ifndef _YOLOX_CPP_YOLOX_TENSORRT_HPP
#define _YOLOX_CPP_YOLOX_TENSORRT_HPP

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "core.hpp"
#include "coco_names.hpp"
#include "tensorrt_logging.h"

namespace yolox_cpp{
    using namespace nvinfer1;

    #define CHECK(status) \
        do\
        {\
            auto ret = (status);\
            if (ret != 0)\
            {\
                std::cerr << "CUDA Failure: " << ret << std::endl;\
                abort();\
            }\
        } while (0)


    class YoloXTensorRT: public AbcYoloX{
        public:
            YoloXTensorRT(const file_name_t &path_to_engine, int device=0,
                          float nms_th=0.45, float conf_th=0.3, const std::string &model_version="0.1.1rc0",
                          int num_classes=80, bool p6=false);
            ~YoloXTensorRT();
            std::vector<Object> inference(const cv::Mat &frame, uchar3* d_image, float* d_output, 
                                            cudaStream_t copy_stream_, cudaStream_t resize_stream_);
        private:
            void doInference(const float* input, float* output, cudaStream_t copy_stream_);

            int DEVICE_ = 0;
            Logger gLogger_;
            std::unique_ptr<IRuntime> runtime_;
            std::unique_ptr<ICudaEngine> engine_;
            std::unique_ptr<IExecutionContext> context_;
            int output_size_;
            const int inputIndex_ = 0;
            const int outputIndex_ = 1;
            void *inference_buffers_[2];
            std::vector<float> input_blob_;
            std::vector<float> output_blob_;

    };
} // namespace yolox_cpp

#endif
