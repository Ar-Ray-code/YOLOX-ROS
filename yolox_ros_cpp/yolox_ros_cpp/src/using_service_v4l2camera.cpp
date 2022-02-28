#include "yolox_ros_cpp/using_service_v4l2camera.hpp"

using namespace std::chrono_literals;
using namespace std::placeholders;

namespace using_service_v4l2camera
{

    void using_service::yolox_callback()
    {
        RCLCPP_INFO(this->get_logger(), "yolox_callback");
        while (!client_yolox->wait_for_service(1s))
        {
            if (!rclcpp::ok())
            {
                RCLCPP_INFO(this->get_logger(), "Client interrupted while waiting for service");
                return ;
            }
            RCLCPP_INFO(this->get_logger(), "waiting for service...");
        }
        

        request = std::make_shared<yolo_msgs::srv::DetectObject::Request>();
        request->image = *image_msg;

        future_yolox = client_yolox->async_send_request(request, std::bind(&using_service::callback_response,this,_1));
    }

    void using_service::callback_response(rclcpp::Client<yolo_msgs::srv::DetectObject>::SharedFuture future) {
        auto response = future.get();
        RCLCPP_INFO(this->get_logger(), "callback_response");

        std::vector<yolo_msgs::msg::BoundingBox> boundingboxes;
        for (auto &box : response->bounding_boxes)
        {
            yolo_msgs::msg::BoundingBox boundingbox;
            boundingbox.xmin = box.xmin;
            boundingbox.ymin = box.ymin;
            boundingbox.xmax = box.xmax;
            boundingbox.ymax = box.ymax;
            boundingbox.class_id = box.class_id;
            boundingbox.confidence = box.confidence;
            boundingboxes.push_back(boundingbox);
        }

        // print all
        // for (auto &bbox : response->bounding_boxes)
        // {
        //     RCLCPP_INFO(this->get_logger(), "xmin: %d, ymin: %d, xmax: %d, ymax: %d, class_id: %s, confidence: %f",
        //                 bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.class_id.c_str(), bbox.confidence);
        // 
        // }

        // Publish bboxes
        yolo_msgs::msg::BoundingBoxes boundingboxes_msg;
        boundingboxes_msg.bounding_boxes = boundingboxes;
        pub_boundingboxes->publish(boundingboxes_msg);

        // draw boundingboxes
        for (auto &bbox : boundingboxes)
        {
            cv::rectangle(frame, cv::Point(bbox.xmin, bbox.ymin), cv::Point(bbox.xmax, bbox.ymax), cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow("frame", frame);
        cv::waitKey(1);
    }

    // Subscription
    void using_service::callback_image(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        frame = cv_ptr->image;
        image_msg = msg;
        yolox_callback();
    }

    using_service::using_service(const rclcpp::NodeOptions& options): Node("using_service", options)
    {
        client_yolox = this->create_client<yolo_msgs::srv::DetectObject>("detect_object");
        sub_image = this->create_subscription<sensor_msgs::msg::Image>("image_raw", 10, std::bind(&using_service::callback_image, this, _1));
        pub_boundingboxes = this->create_publisher<yolo_msgs::msg::BoundingBoxes>("boundingboxes", 10);
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(using_service_v4l2camera::using_service)



// int main(int argc, char** argv)
// {
//     rclcpp::init(argc, argv);
//     rclcpp::NodeOptions options;
//     auto node = std::make_shared<using_service>("using_service",options);

//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }