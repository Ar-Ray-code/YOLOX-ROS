#include "yolox_ros_cpp/yolox_node.hpp"

namespace yolox_ros_cpp{

    YoloXNode::YoloXNode(const rclcpp::NodeOptions& options)
    : YoloXNode::YoloXNode("", options)
    {}

    YoloXNode::YoloXNode(const std::string &node_name, const rclcpp::NodeOptions& options)
    : rclcpp::Node("yolox_ros_cpp", node_name, options){
        
        RCLCPP_INFO(this->get_logger(), "initialize");
        this->initializeParameter();

        if(this->imshow_){
            cv::namedWindow(this->WINDOW_NAME_, cv::WINDOW_AUTOSIZE);
        }

        this->yolox_ = std::make_unique<YoloX>(this->model_path_, this->device_,
                                               this->nms_th_, this->conf_th_, 
                                               this->image_width_, this->image_height_);

        auto qos_img = rclcpp::QoS(
            rclcpp::QoSInitialization(
                rmw_qos_profile_default.history, //rmw_qos_profile_default.reliability
                rmw_qos_profile_default.depth
        ));
        qos_img.reliability(rmw_qos_profile_default.reliability);
        this->sub_image_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image_raw",
            qos_img, 
            std::bind(&YoloXNode::colorImageCallback, this, std::placeholders::_1)
        );
        this->pub_bboxes_ = this->create_publisher<bboxes_ex_msgs::msg::BoundingBoxes>(
            "yolox/bounding_boxes",
            10
        );
        this->pub_image_ = image_transport::create_publisher(this, "yolox/image_raw");

    }

    void YoloXNode::initializeParameter(){
        this->declare_parameter("imshow_isshow", true);
        this->declare_parameter("model_path", "/home/ubuntu/ros2_ws/build/weights/openvino/yolox_nano.xml");
        this->declare_parameter("conf", 0.3f);
        this->declare_parameter("nms", 0.45f);
        this->declare_parameter("device", "CPU");
        this->declare_parameter("image_size/width", 416);
        this->declare_parameter("image_size/height", 416);
        this->get_parameter("imshow_isshow", this->imshow_);
        this->get_parameter("model_path", this->model_path_);
        this->get_parameter("conf", this->conf_th_);
        this->get_parameter("nms", this->nms_th_);
        this->get_parameter("device", this->device_);
        this->get_parameter("image_size/width", this->image_width_);
        this->get_parameter("image_size/height", this->image_height_);

    }
    void YoloXNode::colorImageCallback(const sensor_msgs::msg::Image::SharedPtr ptr){
        auto img = cv_bridge::toCvCopy(ptr, "bgr8");
        cv::Mat frame = img->image;

        auto objects = this->yolox_->inference(frame);
        draw_objects(frame, objects);
        if(this->imshow_){
            cv::imshow(this->WINDOW_NAME_, frame);
            auto key = cv::waitKey(1);
        }

        auto boxes = objects_to_bboxes(frame, objects, img->header);
        this->pub_bboxes_->publish(boxes);

        sensor_msgs::msg::Image::SharedPtr pub_img;
        pub_img = cv_bridge::CvImage(img->header, "bgr8", frame).toImageMsg();
        this->pub_image_.publish(pub_img);
    }
    bboxes_ex_msgs::msg::BoundingBoxes YoloXNode::objects_to_bboxes(cv::Mat frame, std::vector<Object> objects,std_msgs::msg::Header header){
        bboxes_ex_msgs::msg::BoundingBoxes boxes;
        boxes.header = header;
        for(auto obj: objects){
            bboxes_ex_msgs::msg::BoundingBox box;
            box.probability = obj.prob;
            box.class_id = COCO_CLASSES[obj.label];
            box.xmin = obj.rect.x;
            box.ymin = obj.rect.y;
            box.xmax = (obj.rect.x + obj.rect.width);
            box.ymax = (obj.rect.y + obj.rect.height);
            box.img_width = frame.cols;
            box.img_height = frame.rows;
            // tracking id
            // box.id = 0;
            // depth
            // box.center_dist = 0;
            boxes.bounding_boxes.emplace_back(box);
        }
        return boxes;
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(yolox_ros_cpp::YoloXNode)

