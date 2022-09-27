#ifndef _YOLOX_CPP_YOLOX_ONNX_HPP
#define _YOLOX_CPP_YOLOX_ONNX_HPP

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#include "core.hpp"
#include "coco_names.hpp"

namespace yolox_cpp{
    class YoloXONNXRuntime: public AbcYoloX{
        public:
            YoloXONNXRuntime(file_name_t path_to_model,
                             int intra_num_threads, int inter_num_threadsint=1,
                             std::string device="cuda", int device_id=0,
                             float nms_th=0.45, float conf_th=0.3, std::string model_version="0.1.1rc0");
            std::vector<Object> inference(const cv::Mat& frame) override;

        private:
            int inter_num_threads_ = 1;
            int intra_num_threads_ = 1;
            int device_id_ = 0;
            std::string device_;

            Ort::Session session_{nullptr};
            Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "Default"};

            Ort::Value input_tensor_{nullptr};
            Ort::Value output_tensor_{nullptr};
            std::string input_name_;
            std::string output_name_;
            std::vector<std::unique_ptr<uint8_t[]>> input_buffer_;
            std::vector<std::unique_ptr<uint8_t[]>> output_buffer_;
    };
}

#endif
