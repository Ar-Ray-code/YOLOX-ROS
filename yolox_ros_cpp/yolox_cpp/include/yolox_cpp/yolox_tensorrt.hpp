#ifndef _YOLOX_CPP_YOLOX_TENSORRT_HPP
#define _YOLOX_CPP_YOLOX_TENSORRT_HPP

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "cuda_runtime_api.h"
#include "NvInfer.h"

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
                std::cerr << "Cuda failure: " << ret << std::endl;\
                abort();\
            }\
        } while (0)


    class YoloXTensorRT: public AbcYoloX{
        public:
            YoloXTensorRT(file_name_t path_to_engine, int device=0,
                          float nms_th=0.45, float conf_th=0.3, std::string model_version="0.1.1rc0",
                          int num_classes=80, bool p6=false);
            std::vector<Object> inference(const cv::Mat& frame) override;

        private:
            void doInference(float* input, float* output);

            int DEVICE_ = 0;
            Logger gLogger_;
            std::unique_ptr<IRuntime> runtime_;
            std::unique_ptr<ICudaEngine> engine_;
            std::unique_ptr<IExecutionContext> context_;
            int output_size_;
            const int inputIndex_ = 0;
            const int outputIndex_ = 1;

    };
} // namespace yolox_cpp

#endif