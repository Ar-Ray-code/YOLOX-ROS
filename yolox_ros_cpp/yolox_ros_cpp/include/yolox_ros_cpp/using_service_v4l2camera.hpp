#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

#include "yolo_msgs/srv/detect_object.hpp"
#include "yolo_msgs/msg/bounding_box.hpp"
#include "yolo_msgs/msg/bounding_boxes.hpp"

#include <chrono>

namespace using_service_v4l2camera{

    class using_service:public rclcpp::Node
    {   
        private:
            rclcpp::Client<yolo_msgs::srv::DetectObject>::SharedPtr client_yolox;
            rclcpp::Client<yolo_msgs::srv::DetectObject>::SharedFuture future_yolox;

            rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image;
            rclcpp::Publisher<yolo_msgs::msg::BoundingBoxes>::SharedPtr pub_boundingboxes;
            yolo_msgs::srv::DetectObject::Request::SharedPtr request;

            sensor_msgs::msg::Image::SharedPtr image_msg;
            

            // frame cache
            cv::Mat frame;

            void yolox_callback();

            void callback_response(rclcpp::Client<yolo_msgs::srv::DetectObject>::SharedFuture future);

            // Subscription
            void callback_image(const sensor_msgs::msg::Image::SharedPtr msg);

        public:
            using_service(const rclcpp::NodeOptions& options);
    };
    
}