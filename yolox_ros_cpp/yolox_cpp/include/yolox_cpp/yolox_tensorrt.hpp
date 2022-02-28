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


    class YoloXTensorRT: public AbsYoloX{
        public:
            YoloXTensorRT(file_name_t path_to_engine, int device=0,
                          float nms_th=0.45, float conf_th=0.3,
                          int input_width=416, int input_height=416);
            ~YoloXTensorRT();
            std::vector<Object> inference(cv::Mat frame) override;

        private:
            int DEVICE_ = 0;

            Logger gLogger_;
            std::unique_ptr<IRuntime> runtime_;
            std::unique_ptr<ICudaEngine> engine_;
            std::unique_ptr<IExecutionContext> context_;
            int output_size_;
            const int inputIndex_ = 0;
            const int outputIndex_ = 1;

            cv::Mat pr_img_;
            cv::Mat re_;

            const std::vector<float> mean_ = {0.485, 0.456, 0.406};
            const std::vector<float> std_ = {0.229, 0.224, 0.225};

            cv::Mat static_resize(cv::Mat& img);
            void generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);
            float intersection_area(const Object& a, const Object& b);
            void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);
            void qsort_descent_inplace(std::vector<Object>& objects);
            void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);
            void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<Object>& objects);
            float* blobFromImage(cv::Mat& img);
            void decode_outputs(float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h);
            void doInference(float* input, float* output);
    };
} // namespace yolox_cpp

#endif