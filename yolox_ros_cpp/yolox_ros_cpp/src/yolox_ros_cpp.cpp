#include "yolox_ros_cpp/yolox_ros_cpp.hpp"

namespace yolox_ros_cpp{

    YoloXNode::YoloXNode(const rclcpp::NodeOptions& options)
    : YoloXNode::YoloXNode("", options)
    {}

    YoloXNode::YoloXNode(const std::string &node_name, const rclcpp::NodeOptions& options)
    : rclcpp::Node("yolox_ros_cpp", node_name, options){
        
        RCLCPP_INFO(this->get_logger(), "initialize");
        this->initializeParameter();

        if(this->imshow_){
            char window_name[50];
            sprintf(window_name, "%s %s %s", this->WINDOW_NAME_.c_str(), "_", this->get_name());
            this->WINDOW_NAME_ = window_name;            

            cv::namedWindow(this->WINDOW_NAME_, cv::WINDOW_AUTOSIZE);
        }
        
        if(this->model_type_ == "tensorrt"){
            #ifdef YOLOX_USE_TENSORRT
                RCLCPP_INFO(this->get_logger(), "Model Type is TensorRT");
                this->yolox_ = std::make_unique<yolox_cpp::YoloXTensorRT>(this->model_path_, std::stoi(this->device_),
                                                                          this->nms_th_, this->conf_th_, 
                                                                          this->image_width_, this->image_height_,
                                                                          this->INPUT_BLOB_NAME_, this->OUTPUT_BLOB_NAME_);
            #else
                RCLCPP_ERROR(this->get_logger(), "yolox_cpp is not built with TensorRT");
                rclcpp::shutdown();
            #endif
        }else if(this->model_type_ == "openvino"){
            #ifdef YOLOX_USE_OPENVINO
                RCLCPP_INFO(this->get_logger(), "Model Type is OpenVINO");
                this->yolox_ = std::make_unique<yolox_cpp::YoloXOpenVINO>(this->model_path_, this->device_,
                                                                          this->nms_th_, this->conf_th_,
                                                                          this->image_width_, this->image_height_);
            #else
                RCLCPP_ERROR(this->get_logger(), "yolox_cpp is not built with OpenVINO");
                rclcpp::shutdown();
            #endif
        }

        this->sub_image_ = image_transport::create_subscription(
            this, "image_raw", 
            std::bind(&YoloXNode::colorImageCallback, this, std::placeholders::_1), 
            "raw");
        this->pub_bboxes_ = this->create_publisher<bboxes_ex_msgs::msg::BoundingBoxes>(
            "yolox/bounding_boxes",
            10
        );
        this->pub_image_ = image_transport::create_publisher(this, "yolox/image_raw");

    }

    void YoloXNode::initializeParameter(){
        this->declare_parameter("imshow_isshow", true);
        this->declare_parameter("model_path", "/root/ros2_ws/src/YOLOX-ROS/weights/tensorrt/YOLOX_outputs/nano/model_trt.engine");
        this->declare_parameter("conf", 0.3f);
        this->declare_parameter("nms", 0.45f);
        this->declare_parameter("device", "0");
        this->declare_parameter("image_size/width", 416);
        this->declare_parameter("image_size/height", 416);
        this->declare_parameter("model_type", "tensorrt");
        this->declare_parameter("input_blob_name", "input_0");
        this->declare_parameter("output_blob_name", "output_0");

        this->get_parameter("imshow_isshow", this->imshow_);
        this->get_parameter("model_path", this->model_path_);
        this->get_parameter("conf", this->conf_th_);
        this->get_parameter("nms", this->nms_th_);
        this->get_parameter("device", this->device_);
        this->get_parameter("image_size/width", this->image_width_);
        this->get_parameter("image_size/height", this->image_height_);
        this->get_parameter("model_type", this->model_type_);
        this->get_parameter("input_blob_name", this->INPUT_BLOB_NAME_);
        this->get_parameter("output_blob_name", this->OUTPUT_BLOB_NAME_);

        RCLCPP_INFO(this->get_logger(), "Set parameter imshow_isshow: %i", this->imshow_);
        RCLCPP_INFO(this->get_logger(), "Set parameter model_path: '%s'", this->model_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "Set parameter conf: %f", this->conf_th_);
        RCLCPP_INFO(this->get_logger(), "Set parameter nms: %f", this->nms_th_);
        RCLCPP_INFO(this->get_logger(), "Set parameter device: %s", this->device_.c_str());
        RCLCPP_INFO(this->get_logger(), "Set parameter image_size/width: %i", this->image_width_);
        RCLCPP_INFO(this->get_logger(), "Set parameter image_size/height: %i", this->image_height_);
        RCLCPP_INFO(this->get_logger(), "Set parameter model_type: '%s'", this->model_type_.c_str());
        RCLCPP_INFO(this->get_logger(), "Set parameter input_blob_name: '%s'", this->INPUT_BLOB_NAME_.c_str());
        RCLCPP_INFO(this->get_logger(), "Set parameter output_blob_name: '%s'", this->OUTPUT_BLOB_NAME_.c_str());

    }
    void YoloXNode::colorImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& ptr){
        auto img = cv_bridge::toCvCopy(ptr, "bgr8");
        cv::Mat frame = img->image;

        int64 start = cv::getTickCount();
        auto objects = this->yolox_->inference(frame);
        double elapsed_ms = (cv::getTickCount() - start) * 1000 / cv::getTickFrequency();
        double fps = 1000.0 / elapsed_ms; 
        RCLCPP_DEBUG(this->get_logger(), "YOLOX Inference: %lf FPS", fps);

        yolox_cpp::utils::draw_objects(frame, objects);
        if(this->imshow_){
            cv::imshow(this->WINDOW_NAME_, frame);
            auto key = cv::waitKey(1);
            if(key == 27){
                rclcpp::shutdown();
            }
        }

        auto boxes = objects_to_bboxes(frame, objects, img->header);
        this->pub_bboxes_->publish(boxes);

        sensor_msgs::msg::Image::SharedPtr pub_img;
        pub_img = cv_bridge::CvImage(img->header, "bgr8", frame).toImageMsg();
        this->pub_image_.publish(pub_img);
    }
    bboxes_ex_msgs::msg::BoundingBoxes YoloXNode::objects_to_bboxes(cv::Mat frame, std::vector<yolox_cpp::Object> objects, std_msgs::msg::Header header){
        bboxes_ex_msgs::msg::BoundingBoxes boxes;
        boxes.header = header;
        for(auto obj: objects){
            bboxes_ex_msgs::msg::BoundingBox box;
            box.probability = obj.prob;
            box.class_id = yolox_cpp::COCO_CLASSES[obj.label];
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

