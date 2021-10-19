#ifndef _YOLOX_NODE_HPP
#define _YOLOX_NODE_HPP
#include <math.h>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
// #include <ament_index_cpp/get_package_share_directory.hpp>

#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

#include "bboxes_ex_msgs/msg/bounding_box.hpp"
#include "bboxes_ex_msgs/msg/bounding_boxes.hpp"

#include "yolox_ros_cpp/yolox_openvino.hpp"
#include "yolox_ros_cpp/utils.hpp"

using namespace yolox_ros_cpp;
using namespace yolox_ros_cpp::utils;
using namespace yolox_ros_cpp::openvino;

namespace yolox_ros_cpp{

    class YoloXNode : public rclcpp::Node
    {
    public:
        YoloXNode(const rclcpp::NodeOptions& options);
        YoloXNode(const std::string &node_name, const rclcpp::NodeOptions& options);

    private:
        void initializeParameter();
        std::unique_ptr<YoloX> yolox_;
        std::string model_path_;
        std::string device_;
        float conf_th_;
        float nms_th_;
        int image_width_;
        int image_height_;

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
        void colorImageCallback(const sensor_msgs::msg::Image::SharedPtr ptr);

        rclcpp::Publisher<bboxes_ex_msgs::msg::BoundingBoxes>::SharedPtr pub_bboxes_;
        image_transport::Publisher pub_image_;

        bboxes_ex_msgs::msg::BoundingBoxes objects_to_bboxes(cv::Mat frame, std::vector<Object> objects,std_msgs::msg::Header header);

        const std::string WINDOW_NAME_ = "YOLOX";
        bool imshow_ = false;
    };
}
#endif