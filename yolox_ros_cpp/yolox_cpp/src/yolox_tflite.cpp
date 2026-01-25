#include "yolox_cpp/yolox_tflite.hpp"

namespace yolox_cpp {

YoloXTflite::YoloXTflite(const file_name_t &path_to_model, int num_threads,
                         float nms_th, float conf_th,
                         const std::string &model_version, int num_classes,
                         bool p6, bool is_nchw)
    : AbcYoloX(nms_th, conf_th, model_version, num_classes, p6),
      is_nchw_(is_nchw) {
    TfLiteStatus status;
    this->model_ =
        tflite::FlatBufferModel::BuildFromFile(path_to_model.c_str());
    TFLITE_MINIMAL_CHECK(model_);

    this->resolver_ =
        std::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
    this->interpreter_ = std::make_unique<tflite::Interpreter>();

    tflite::InterpreterBuilder builder(*model_, *this->resolver_);
    builder(&this->interpreter_);
    TFLITE_MINIMAL_CHECK(this->interpreter_ != nullptr);

    TFLITE_MINIMAL_CHECK(this->interpreter_->AllocateTensors() == kTfLiteOk);
    // tflite::PrintInterpreterState(this->interpreter_.get());

    status = this->interpreter_->SetNumThreads(num_threads);
    if (status != TfLiteStatus::kTfLiteOk) {
        std::string msg = "Failed to SetNumThreads.";
        throw std::runtime_error(msg.c_str());
    }

    // XNNPACK Delegate
    auto xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_options.num_threads = num_threads;
    this->delegate_ = TfLiteXNNPackDelegateCreate(&xnnpack_options);
    status = this->interpreter_->ModifyGraphWithDelegate(this->delegate_);
    if (status != TfLiteStatus::kTfLiteOk) {
        std::string msg = "Failed to ModifyGraphWithDelegate.";
        throw std::runtime_error(msg.c_str());
    }

    // // GPU Delegate
    // auto gpu_options = TfLiteGpuDelegateOptionsV2Default();
    // gpu_options.inference_preference =
    // TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    // gpu_options.inference_priority1 =
    // TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY; this->delegate_ =
    // TfLiteGpuDelegateV2Create(&gpu_options); status =
    // this->interpreter_->ModifyGraphWithDelegate(this->delegate_); if (status
    // != TfLiteStatus::kTfLiteOk)
    // {
    //     std::cerr << "Failed to ModifyGraphWithDelegate." << std::endl;
    //     exit(1);
    // }

    // // NNAPI Delegate
    // tflite::StatefulNnApiDelegate::Options nnapi_options;
    // nnapi_options.execution_preference =
    // tflite::StatefulNnApiDelegate::Options::kSustainedSpeed;
    // nnapi_options.allow_fp16 = true;
    // nnapi_options.disallow_nnapi_cpu = true;
    // this->delegate_ = new tflite::StatefulNnApiDelegate(nnapi_options);
    // status = this->interpreter_->ModifyGraphWithDelegate(this->delegate_);
    // if (status != TfLiteStatus::kTfLiteOk)
    // {
    //     std::cerr << "Failed to ModifyGraphWithDelegate." << std::endl;
    //     exit(1);
    // }

    if (this->interpreter_->AllocateTensors() != TfLiteStatus::kTfLiteOk) {
        std::string msg = "Failed to allocate tensors.";
        throw std::runtime_error(msg.c_str());
    }

    {
        TfLiteTensor *tensor = this->interpreter_->input_tensor(0);
        std::cout << "input:" << std::endl;
        std::cout << " name: " << tensor->name << std::endl;
        if (this->is_nchw_ == true) {
            // NCHW
            this->input_h_ = tensor->dims->data[2];
            this->input_w_ = tensor->dims->data[3];
        } else {
            // NHWC
            this->input_h_ = tensor->dims->data[1];
            this->input_w_ = tensor->dims->data[2];
        }

        std::cout << " shape:" << std::endl;
        if (tensor->type == kTfLiteUInt8) {
            this->input_size_ = sizeof(uint8_t);
        } else {
            this->input_size_ = sizeof(float);
        }
        for (int i = 0; i < tensor->dims->size; i++) {
            this->input_size_ *= tensor->dims->data[i];
            std::cout << "   - " << tensor->dims->data[i] << std::endl;
        }
        std::cout << " input_h: " << this->input_h_ << std::endl;
        std::cout << " input_w: " << this->input_w_ << std::endl;
        std::cout << " tensor_type: " << tensor->type << std::endl;
    }

    {
        TfLiteTensor *tensor = this->interpreter_->output_tensor(0);
        std::cout << "output:" << std::endl;
        std::cout << " name: " << tensor->name << std::endl;
        std::cout << " shape:" << std::endl;
        if (tensor->type == kTfLiteUInt8) {
            this->output_size_ = sizeof(uint8_t);
        } else {
            this->output_size_ = sizeof(float);
        }
        for (int i = 0; i < tensor->dims->size; i++) {
            this->output_size_ *= tensor->dims->data[i];
            std::cout << "   - " << tensor->dims->data[i] << std::endl;
        }
        std::cout << " tensor_type: " << tensor->type << std::endl;
    }

    // Prepare GridAndStrides
    if (this->p6_) {
        generate_grids_and_stride(this->input_w_, this->input_h_,
                                  this->strides_p6_, this->grid_strides_);
    } else {
        generate_grids_and_stride(this->input_w_, this->input_h_,
                                  this->strides_, this->grid_strides_);
    }
}
YoloXTflite::~YoloXTflite() { TfLiteXNNPackDelegateDelete(this->delegate_); }
std::vector<Object> YoloXTflite::inference(const cv::Mat &frame) {
    // preprocess
    cv::Mat pr_img = static_resize(frame);

    float *input_blob = this->interpreter_->typed_input_tensor<float>(0);
    if (this->is_nchw_ == true) {
        blobFromImage(pr_img, input_blob);
    } else {
        blobFromImage_nhwc(pr_img, input_blob);
    }

    // inference
    TfLiteStatus ret = this->interpreter_->Invoke();
    if (ret != TfLiteStatus::kTfLiteOk) {
        std::cerr << "Failed to invoke." << std::endl;
        return std::vector<Object>();
    }

    // postprocess
    const float scale = std::min(
        static_cast<float>(this->input_w_) / static_cast<float>(frame.cols),
        static_cast<float>(this->input_h_) / static_cast<float>(frame.rows));
    std::vector<Object> objects;
    decode_outputs(this->interpreter_->typed_output_tensor<float>(0),
                   this->grid_strides_, objects, this->bbox_conf_thresh_, scale,
                   frame.cols, frame.rows);

    return objects;
}

} // namespace yolox_cpp
