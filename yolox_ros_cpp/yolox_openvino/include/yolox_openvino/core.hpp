#ifndef _YOLOX_OPENVINO_CORE_HPP
#define _YOLOX_OPENVINO_CORE_HPP

#include <opencv2/core/types.hpp>

namespace yolox_openvino{
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
}
#endif