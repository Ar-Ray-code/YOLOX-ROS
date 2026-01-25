#include "yolox_ros_cpp/yolox_ros_cpp.hpp"

namespace yolox_ros_cpp {
YoloXNode::YoloXNode(const rclcpp::NodeOptions &options)
    : Node("yolox_ros_cpp", options) {
    using namespace std::chrono_literals; // NOLINT
    this->init_timer_ =
        this->create_wall_timer(0s, std::bind(&YoloXNode::onInit, this));
}

void YoloXNode::onInit() {
    this->init = true;
    this->d_image_ = nullptr;
    this->d_output_ = nullptr;
    cudaStreamCreate(&this->copy_stream_);
    cudaStreamCreate(&this->resize_stream_);
    rclcpp::on_shutdown([this]() {
        cudaStreamDestroy(this->copy_stream_);
        cudaStreamDestroy(this->resize_stream_);
        cudaFree(this->d_image_);
        cudaFree(this->d_output_);
        this->copy_stream_ = nullptr;
        this->resize_stream_ = nullptr;
    });
    this->init_timer_->cancel();
    this->param_listener_ = std::make_shared<yolox_parameters::ParamListener>(
        this->get_node_parameters_interface());
    this->params_ = this->param_listener_->get_params();
    this->callback_group_reentrant_ =
        this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    this->sub_options_ = std::make_shared<rclcpp::SubscriptionOptions>();
    this->sub_options_->callback_group = callback_group_reentrant_;
    this->sub_options_->use_intra_process_comm =
        rclcpp::IntraProcessSetting::Enable;
    if (this->params_.imshow_isshow) {
        cv::namedWindow("yolox", cv::WINDOW_AUTOSIZE);
    }

    if (this->params_.class_labels_path != "") {
        RCLCPP_INFO(this->get_logger(), "read class labels from '%s'",
                    this->params_.class_labels_path.c_str());
        this->class_names_ = yolox_cpp::utils::read_class_labels_file(
            this->params_.class_labels_path);
    } else {
        this->class_names_ = yolox_cpp::COCO_CLASSES;
    }

    if (this->params_.model_type == "tensorrt") {
#ifdef ENABLE_TENSORRT
        RCLCPP_INFO(this->get_logger(), "Model Type is TensorRT");
        this->yolox_ = std::make_unique<yolox_cpp::YoloXTensorRT>(
            this->params_.model_path, this->params_.tensorrt_device,
            this->params_.nms, this->params_.conf, this->params_.model_version,
            this->params_.num_classes, this->params_.p6);
#else
        RCLCPP_ERROR(this->get_logger(),
                     "yolox_cpp is not built with TensorRT");
        rclcpp::shutdown();
#endif
    } else if (this->params_.model_type == "openvino") {
#ifdef ENABLE_OPENVINO
        RCLCPP_INFO(this->get_logger(), "Model Type is OpenVINO");
        this->yolox_ = std::make_unique<yolox_cpp::YoloXOpenVINO>(
            this->params_.model_path, this->params_.openvino_device,
            this->params_.nms, this->params_.conf, this->params_.model_version,
            this->params_.num_classes, this->params_.p6);
#else
        RCLCPP_ERROR(this->get_logger(),
                     "yolox_cpp is not built with OpenVINO");
        rclcpp::shutdown();
#endif
    } else if (this->params_.model_type == "onnxruntime") {
#ifdef ENABLE_ONNXRUNTIME
        RCLCPP_INFO(this->get_logger(), "Model Type is ONNXRuntime");
        this->yolox_ = std::make_unique<yolox_cpp::YoloXONNXRuntime>(
            this->params_.model_path,
            this->params_.onnxruntime_intra_op_num_threads,
            this->params_.onnxruntime_inter_op_num_threads,
            this->params_.onnxruntime_use_cuda,
            this->params_.onnxruntime_device_id,
            this->params_.onnxruntime_use_parallel, this->params_.nms,
            this->params_.conf, this->params_.model_version,
            this->params_.num_classes, this->params_.p6);
#else
        RCLCPP_ERROR(this->get_logger(),
                     "yolox_cpp is not built with ONNXRuntime");
        rclcpp::shutdown();
#endif
    } else if (this->params_.model_type == "tflite") {
#ifdef ENABLE_TFLITE
        RCLCPP_INFO(this->get_logger(), "Model Type is tflite");
        this->yolox_ = std::make_unique<yolox_cpp::YoloXTflite>(
            this->params_.model_path, this->params_.tflite_num_threads,
            this->params_.nms, this->params_.conf, this->params_.model_version,
            this->params_.num_classes, this->params_.p6, this->params_.is_nchw);
#else
        RCLCPP_ERROR(this->get_logger(), "yolox_cpp is not built with tflite");
        rclcpp::shutdown();
#endif
    }
    RCLCPP_INFO(this->get_logger(), "model loaded");

    rclcpp::QoS qos_profile(5); // Queue depth of 5 for multithreading
    this->sub_image_ = image_transport::create_subscription(
        this, this->params_.src_image_topic_name,
        std::bind(&YoloXNode::colorImageCallback, this, std::placeholders::_1),
        "raw", qos_profile.get_rmw_qos_profile(), *this->sub_options_);

    if (this->params_.use_bbox_ex_msgs) {
        this->pub_bboxes_ =
            this->create_publisher<bboxes_ex_msgs::msg::BoundingBoxes>(
                this->params_.publish_boundingbox_topic_name, 10);
    } else {
        this->pub_detection2d_ =
            this->create_publisher<tr_messages::msg::DetWithImg>(
                this->params_.publish_boundingbox_topic_name, 10);
    }

    if (this->params_.publish_resized_image) {
        this->pub_image_ = image_transport::create_publisher(
            this, this->params_.publish_image_topic_name);
    }
}

void YoloXNode::colorImageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr &ptr) {
    auto now_noninf = std::chrono::system_clock::now();
    auto img = cv_bridge::toCvShare(ptr, "bgr8");

    auto now = std::chrono::system_clock::now();
    // Initialization
    if (this->init) {
        size_t image_bytes = sizeof(uchar3) * img->image.cols * img->image.rows;
        cudaMallocManaged(reinterpret_cast<void **>(&this->d_image_),
                          image_bytes);
        size_t output_bytes = sizeof(float) * 416 * 416 * 3;
        cudaMallocManaged(reinterpret_cast<void **>(&this->d_output_),
                          output_bytes);
        this->init = false;
    }
    auto objects = this->yolox_->inference(
        img->image, this->d_image_, this->d_output_, this->copy_stream_,
        this->resize_stream_); // Use img->image
    auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - now);

    if (this->params_.imshow_isshow) {
        yolox_cpp::utils::draw_objects(img->image, objects, this->class_names_);
        cv::imshow("yolox", img->image);
        if (cv::waitKey(1) == 27) {
            rclcpp::shutdown();
        }
    }

    if (this->params_.use_bbox_ex_msgs) {
        if (!this->pub_bboxes_) {
            RCLCPP_ERROR(this->get_logger(), "pub_bboxes_ is nullptr");
            return;
        }
        auto boxes =
            objects_to_bboxes(img->image, objects, img->header); // Fix here
        this->pub_bboxes_->publish(boxes);
    } else {
        if (!this->pub_detection2d_) {
            RCLCPP_ERROR(this->get_logger(), "pub_detection2d_ is nullptr");
            return;
        }
        vision_msgs::msg::Detection2DArray detections =
            objects_to_detection2d(objects, img->header);
        if (!detections.detections.empty()) {
            tr_messages::msg::DetWithImg detwithimg;
            // detwithimg.image = *ptr;  // Copy unavoidable due to const shared
            // ptr
            detwithimg.detection_info.detections = detections.detections;
            this->pub_detection2d_->publish(detwithimg);
        } else {
            RCLCPP_INFO(this->get_logger(), "no detections so not publishing");
        }
    }
    auto end_noninf = std::chrono::system_clock::now();
    auto elapsed_noninf = std::chrono::duration_cast<std::chrono::microseconds>(
        end_noninf - now_noninf);
    RCLCPP_INFO(this->get_logger(),
                "Inference time: %5ld us Non Inference time: %5ld",
                elapsed.count(), elapsed_noninf.count() - elapsed.count());
}

bboxes_ex_msgs::msg::BoundingBoxes
YoloXNode::objects_to_bboxes(const cv::Mat &frame,
                             const std::vector<yolox_cpp::Object> &objects,
                             const std_msgs::msg::Header &header) {
    bboxes_ex_msgs::msg::BoundingBoxes boxes;
    boxes.header = header;
    for (const auto &obj : objects) {
        bboxes_ex_msgs::msg::BoundingBox box;
        box.probability = obj.prob;
        ;
        box.class_id = std::to_string(obj.label);
        box.xmin = obj.rect.x;
        box.ymin = obj.rect.y;
        box.xmax = (obj.rect.x + obj.rect.width);
        box.ymax = (obj.rect.y + obj.rect.height);
        box.img_width = frame.cols;
        box.img_height = frame.rows;
        boxes.bounding_boxes.emplace_back(box);
    }
    return boxes;
}

vision_msgs::msg::Detection2DArray
YoloXNode::objects_to_detection2d(const std::vector<yolox_cpp::Object> &objects,
                                  const std_msgs::msg::Header &header) {
    vision_msgs::msg::Detection2DArray detection2d;
    detection2d.header = header;
    for (const auto &obj : objects) {
        vision_msgs::msg::Detection2D det;
        det.bbox.center.position.x = obj.rect.x + obj.rect.width / 2;
        det.bbox.center.position.y = obj.rect.y + obj.rect.height / 2;
        det.bbox.size_x = obj.rect.width;
        det.bbox.size_y = obj.rect.height;

        det.results.resize(1);
        det.results[0].hypothesis.class_id = std::to_string(obj.label);
        det.results[0].hypothesis.score = obj.prob;
        detection2d.detections.emplace_back(det);
    }
    return detection2d;
}
} // namespace yolox_ros_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolox_ros_cpp::YoloXNode)
