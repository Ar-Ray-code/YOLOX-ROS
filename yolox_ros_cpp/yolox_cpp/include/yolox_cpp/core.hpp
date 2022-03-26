#ifndef _YOLOX_CPP_CORE_HPP
#define _YOLOX_CPP_CORE_HPP

#include <opencv2/core/types.hpp>

namespace yolox_cpp{
    /**
     * @brief Define names based depends on Unicode path support
     */
    #define tcout                  std::cout
    #define file_name_t            std::string
    #define imread_t               cv::imread

    struct Object
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
    };

    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
    };
    
    class AbcYoloX{
        public:
            AbcYoloX(){}
            AbcYoloX(float nms_th=0.45, float conf_th=0.3,
                     int input_width=416, int input_height=416)
            :nms_thresh_(nms_th), bbox_conf_thresh_(conf_th),
             input_w_(input_width), input_h_(input_height)
            {}
            virtual std::vector<Object> inference(cv::Mat frame) = 0;
        protected:
            int input_w_;
            int input_h_;
            float nms_thresh_;
            float bbox_conf_thresh_;
            int num_classes_ = 80;
    };
}
#endif