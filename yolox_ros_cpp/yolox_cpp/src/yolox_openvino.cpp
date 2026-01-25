#include "yolox_cpp/yolox_openvino.hpp"

namespace yolox_cpp {
YoloXOpenVINO::YoloXOpenVINO(const file_name_t &path_to_model,
                             std::string device_name, float nms_th,
                             float conf_th, const std::string &model_version,
                             int num_classes, bool p6)
    : AbcYoloX(nms_th, conf_th, model_version, num_classes, p6),
      device_name_(device_name) {
    // Step 1. Initialize inference engine core
    std::cout << "Initialize Inference engine core" << std::endl;
    ov::Core ie;

    // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and
    // .bin files) or ONNX (.onnx file) format
    std::cout << "Read a model in OpenVINO Intermediate Representation: "
              << path_to_model << std::endl;
    const auto network = ie.read_model(path_to_model);

    //  Step 3. Loading a model to the device
    std::vector<std::string> available_devices = ie.get_available_devices();
    std::cout << "======= AVAILABLE DEVICES FOR OPENVINO =======" << std::endl;
    for (auto device : available_devices) {
        std::cout << "- " << device << std::endl;
    }
    std::cout << "==============================================" << std::endl;
    std::cout << "Loading a model to the device: " << device_name_ << std::endl;
    auto compiled_model = ie.compile_model(network, device_name);

    // Step 4. Create an infer request
    std::cout << "Create an infer request" << std::endl;
    this->infer_request_ = compiled_model.create_infer_request();

    // Step 5. Configure input & output
    std::cout << "Configuring input and output blobs" << std::endl;
    this->input_shape_ = compiled_model.input(0).get_shape();
    /* Mark input as resizable by setting of a resize algorithm.
     * In this case we will be able to set an input blob of any shape to an
     * infer request. Resize and layout conversions are executed automatically
     * during inference */
    this->blob_.resize(this->input_shape_.at(0) * this->input_shape_.at(1) *
                       this->input_shape_.at(2) * this->input_shape_.at(3));
    this->input_h_ = this->input_shape_.at(2);
    this->input_w_ = this->input_shape_.at(3);
    std::cout << "INPUT_HEIGHT: " << this->input_h_ << std::endl;
    std::cout << "INPUT_WIDTH: " << this->input_w_ << std::endl;

    // Prepare GridAndStrides
    if (this->p6_) {
        generate_grids_and_stride(this->input_w_, this->input_h_,
                                  this->strides_p6_, this->grid_strides_);
    } else {
        generate_grids_and_stride(this->input_w_, this->input_h_,
                                  this->strides_, this->grid_strides_);
    }
}

std::vector<Object> YoloXOpenVINO::inference(const cv::Mat &frame) {
    // preprocess
    cv::Mat pr_img = static_resize(frame);
    // locked memory holder should be alive all time while access to its buffer
    // happens
    blobFromImage(pr_img, this->blob_.data());

    // do inference
    /* Running the request synchronously */
    this->infer_request_.set_input_tensor(
        ov::Tensor{ov::element::f32, this->input_shape_,
                   reinterpret_cast<float *>(this->blob_.data())});
    infer_request_.infer();

    const auto &output_tensor = this->infer_request_.get_output_tensor();
    const float *net_pred = reinterpret_cast<float *>(output_tensor.data());

    const float scale = std::min(
        static_cast<float>(this->input_w_) / static_cast<float>(frame.cols),
        static_cast<float>(this->input_h_) / static_cast<float>(frame.rows));

    std::vector<Object> objects;
    decode_outputs(net_pred, this->grid_strides_, objects,
                   this->bbox_conf_thresh_, scale, frame.cols, frame.rows);
    return objects;
}
} // namespace yolox_cpp
