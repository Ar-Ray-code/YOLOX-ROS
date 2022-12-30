#ifndef _YOLOX_CPP_YOLOX_OPENVINO_HPP
#define _YOLOX_CPP_YOLOX_OPENVINO_HPP

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include "core.hpp"
#include "coco_names.hpp"

namespace yolox_cpp{
    class YoloXOpenVINO: public AbcYoloX{
        public:
            YoloXOpenVINO(file_name_t path_to_model, std::string device_name,
                          float nms_th=0.45, float conf_th=0.3, std::string model_version="0.1.1rc0",
                          int num_classes=80, bool p6=false);
            std::vector<Object> inference(const cv::Mat& frame) override;

        private:
            std::string device_name_;
            std::string input_name_;
            std::string output_name_;
            InferenceEngine::CNNNetwork network_;
            InferenceEngine::ExecutableNetwork executable_network_;
            InferenceEngine::InferRequest infer_request_;
    };
}

#endif