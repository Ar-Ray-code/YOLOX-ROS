#ifndef _YOLOX_OPENVINO_UTILS_HPP
#define _YOLOX_OPENVINO_UTILS_HPP

#include <opencv2/opencv.hpp>
#include "core.hpp"
#include "coco_names.hpp"

namespace yolox_openvino{
    namespace utils{

        static void draw_objects(cv::Mat bgr, const std::vector<Object>& objects)
        {

            // cv::Mat image = bgr.clone();

            for (size_t i = 0; i < objects.size(); i++)
            {
                const Object& obj = objects[i];

                // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                //         obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

                cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
                float c_mean = cv::mean(color)[0];
                cv::Scalar txt_color;
                if (c_mean > 0.5){
                    txt_color = cv::Scalar(0, 0, 0);
                }else{
                    txt_color = cv::Scalar(255, 255, 255);
                }

                cv::rectangle(bgr, obj.rect, color * 255, 2);

                char text[256];
                sprintf(text, "%s %.1f%%", COCO_CLASSES[obj.label], obj.prob * 100);

                int baseLine = 0;
                cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

                cv::Scalar txt_bk_color = color * 0.7 * 255;

                int x = obj.rect.x;
                int y = obj.rect.y + 1;
                //int y = obj.rect.y - label_size.height - baseLine;
                if (y > bgr.rows)
                    y = bgr.rows;
                //if (x + label_size.width > bgr.cols)
                    //x = bgr.cols - label_size.width;

                cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                            txt_bk_color, -1);

                cv::putText(bgr, text, cv::Point(x, y + label_size.height),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
            }
        }
    }
}
#endif