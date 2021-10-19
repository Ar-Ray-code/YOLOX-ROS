#ifndef _YOLOX_OPENVINO_HPP
#define _YOLOX_OPENVINO_HPP

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <inference_engine.hpp>

#include "core.hpp"
#include "coco_names.hpp"

namespace yolox_ros_cpp{
    namespace openvino{

        class YoloX{
            public:
                YoloX(file_name_t path_to_model, std::string device_name, 
                      float nms_th=0.45, float conf_th=0.3,
                      int input_width=416, int input_height=416);
                std::vector<Object> inference(cv::Mat frame);

            private:
                cv::Mat static_resize(cv::Mat& img);
                void blobFromImage(cv::Mat& img, InferenceEngine::Blob::Ptr& blob);
                void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);
                void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr, float prob_threshold, std::vector<Object>& objects);
                float intersection_area(const Object& a, const Object& b);
                void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);
                void qsort_descent_inplace(std::vector<Object>& objects);
                void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);
                void decode_outputs(const float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h);
                int input_w_ = 416;
                int input_h_ = 416;
                int num_classes_ = 80;
                double nms_thresh_ = 0.45;
                double bbox_conf_thresh_ = 0.3;

                std::string input_name_;
                std::string output_name_;
                InferenceEngine::CNNNetwork network_;
                InferenceEngine::ExecutableNetwork executable_network_;
                InferenceEngine::InferRequest infer_request_;
        };

    }
}
#endif