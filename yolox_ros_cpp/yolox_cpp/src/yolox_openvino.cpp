#include "yolox_cpp/yolox_openvino.hpp"

namespace yolox_cpp{
    using namespace InferenceEngine;

    YoloXOpenVINO::YoloXOpenVINO(file_name_t path_to_model, std::string device_name,
                                 float nms_th, float conf_th, std::string model_version,
                                 int num_classes, bool p6)
    :AbcYoloX(nms_th, conf_th, model_version, num_classes, p6),
     device_name_(device_name)
    {
        // Step 1. Initialize inference engine core
        std::cout << "Initialize Inference engine core" << std::endl;
        Core ie;

        // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and
        // .bin files) or ONNX (.onnx file) format
        std::cout << "Read a model in OpenVINO Intermediate Representation: " << path_to_model << std::endl;
        network_ = ie.ReadNetwork(path_to_model);
        if (network_.getOutputsInfo().size() != 1)
            throw std::logic_error("Sample supports topologies with 1 output only");
        if (network_.getInputsInfo().size() != 1)
            throw std::logic_error("Sample supports topologies with 1 input only");

        // Step 3. Configure input & output
        std::cout << "Configuring input and output blobs" << std::endl;
        // Prepare input blobs
        InputInfo::Ptr input_info = network_.getInputsInfo().begin()->second;
        input_name_ = network_.getInputsInfo().begin()->first;

        /* Mark input as resizable by setting of a resize algorithm.
        * In this case we will be able to set an input blob of any shape to an
        * infer request. Resize and layout conversions are executed automatically
        * during inference */
        //input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        //input_info->setLayout(Layout::NHWC);
        input_info->setPrecision(Precision::FP32);
        auto input_dims = input_info->getInputData()->getDims();
        this->input_h_ = input_dims[2];
        this->input_w_ = input_dims[3];
        std::cout << "INPUT_HEIGHT: " << this->input_h_ << std::endl;
        std::cout << "INPUT_WIDTH: " << this->input_w_ << std::endl;

        // Prepare output blobs
        if (network_.getOutputsInfo().empty()) {
            std::cerr << "Network outputs info is empty" << std::endl;
            throw std :: runtime_error( "Network outputs info is empty" );
        }
        DataPtr output_info = network_.getOutputsInfo().begin()->second;
        output_name_ = network_.getOutputsInfo().begin()->first;

        // output_info->setPrecision(Precision::FP16);
        output_info->setPrecision(Precision::FP32);

        //  Step 4. Loading a model to the device
        std::cout << "Loading a model to the device: " << device_name_ << std::endl;
        executable_network_ = ie.LoadNetwork(network_, device_name_);

        //  Step 5. Create an infer request
        std::cout << "Create an infer request" << std::endl;
        infer_request_ = executable_network_.CreateInferRequest();

        // Prepare GridAndStrides
        if(this->p6_)
        {
            generate_grids_and_stride(this->input_w_, this->input_h_, this->strides_p6_, this->grid_strides_);
        }
        else
        {
            generate_grids_and_stride(this->input_w_, this->input_h_, this->strides_, this->grid_strides_);
        }
    }

    std::vector<Object> YoloXOpenVINO::inference(const cv::Mat& frame)
    {
        // preprocess
        cv::Mat pr_img = static_resize(frame);
        InferenceEngine::Blob::Ptr imgBlob = infer_request_.GetBlob(input_name_);
        InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(imgBlob);
        if (!mblob)
        {
            THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
                << "but by fact we were not able to cast inputBlob to MemoryBlob";
        }
        // locked memory holder should be alive all time while access to its buffer happens
        auto mblobHolder = mblob->wmap();
        float *blob_data = mblobHolder.as<float *>();
        blobFromImage(pr_img, blob_data);

        // do inference
        /* Running the request synchronously */
        infer_request_.Infer();

        // Process output
        const InferenceEngine::Blob::Ptr output_blob = infer_request_.GetBlob(output_name_);
        InferenceEngine::MemoryBlob::CPtr moutput = as<InferenceEngine::MemoryBlob>(output_blob);
        if (!moutput) {
            throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                "but by fact we were not able to cast output to MemoryBlob");
        }

        // locked memory holder should be alive all time while access to its buffer
        // happens
        auto moutputHolder = moutput->rmap();
        const float* net_pred = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type*>();

        float scale = std::min(input_w_ / (frame.cols*1.0), input_h_ / (frame.rows*1.0));
        std::vector<Object> objects;
        decode_outputs(net_pred, this->grid_strides_, objects, this->bbox_conf_thresh_, scale, frame.cols, frame.rows);
        return objects;
    }

}