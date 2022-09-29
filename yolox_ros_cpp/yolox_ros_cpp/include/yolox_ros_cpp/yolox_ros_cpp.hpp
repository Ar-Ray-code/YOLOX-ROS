#ifndef _YOLOX_ROS_CPP_YOLOX_ROS_CPP_HPP
#define _YOLOX_ROS_CPP_YOLOX_ROS_CPP_HPP
#include <math.h>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
// #include <ament_index_cpp/get_package_share_directory.hpp>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

#include "bboxes_ex_msgs/msg/bounding_box.hpp"
#include "bboxes_ex_msgs/msg/bounding_boxes.hpp"

#include "yolox_cpp/yolox.hpp"
#include "yolox_cpp/utils.hpp"

namespace yolox_ros_cpp{

    class YoloXNode : public rclcpp::Node
    {
    public:
        YoloXNode(const rclcpp::NodeOptions& options);
        YoloXNode(const std::string &node_name, const rclcpp::NodeOptions& options);

    private:
        void initializeParameter();
        std::unique_ptr<yolox_cpp::AbcYoloX> yolox_;
        std::string model_path_;
        std::string model_type_;
        std::string model_version_;
        int tensorrt_device_;
        std::string openvino_device_;
        bool onnxruntime_use_cuda_;
        int onnxruntime_device_id_;
        bool onnxruntime_use_parallel_;
        int onnxruntime_intra_op_num_threads_;
        int onnxruntime_inter_op_num_threads_;
        float conf_th_;
        float nms_th_;
        int num_classes_;
        std::vector<std::string> class_names_;
        std::string class_labels_path_;

        std::string src_image_topic_name_;
        std::string publish_image_topic_name_;
        std::string publish_boundingbox_topic_name_;

        image_transport::Subscriber sub_image_;
        void colorImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& ptr);

        rclcpp::Publisher<bboxes_ex_msgs::msg::BoundingBoxes>::SharedPtr pub_bboxes_;
        image_transport::Publisher pub_image_;

        bboxes_ex_msgs::msg::BoundingBoxes objects_to_bboxes(cv::Mat frame, std::vector<yolox_cpp::Object> objects, std_msgs::msg::Header header);

        std::string WINDOW_NAME_ = "YOLOX";
        bool imshow_ = true;
    };
}
#endif
