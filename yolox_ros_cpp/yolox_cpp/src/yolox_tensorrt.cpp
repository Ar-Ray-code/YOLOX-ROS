#include "yolox_cpp/yolox_tensorrt.hpp"

namespace yolox_cpp{

    YoloXTensorRT::YoloXTensorRT(file_name_t path_to_engine, int device,
                                 float nms_th, float conf_th, std::string model_version,
                                 int num_classes, bool p6)
    :AbcYoloX(nms_th, conf_th, model_version, num_classes, p6),
     DEVICE_(device)
    {
        cudaSetDevice(this->DEVICE_);
        // create a model using the API directly and serialize it to a stream
        char *trtModelStream{nullptr};
        size_t size{0};

        std::ifstream file(path_to_engine, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }else{
            std::cerr << "invalid arguments path_to_engine: " << path_to_engine << std::endl;
            return;
        }

        this->runtime_ = std::unique_ptr<IRuntime>(createInferRuntime(this->gLogger_));
        assert(this->runtime_ != nullptr);
        this->engine_ = std::unique_ptr<ICudaEngine>(this->runtime_->deserializeCudaEngine(trtModelStream, size));
        assert(this->engine_ != nullptr);
        this->context_ = std::unique_ptr<IExecutionContext>(this->engine_->createExecutionContext());
        assert(this->context_ != nullptr);
        delete[] trtModelStream;

        auto input_dims = this->engine_->getBindingDimensions(0);
        this->input_h_ = input_dims.d[2];
        this->input_w_ = input_dims.d[3];
        std::cout << "INPUT_HEIGHT: " << this->input_h_ << std::endl;
        std::cout << "INPUT_WIDTH: " << this->input_w_ << std::endl;

        auto out_dims = this->engine_->getBindingDimensions(1);
        this->output_size_ = 1;
        for(int j=0; j<out_dims.nbDims; ++j) {
            this->output_size_ *= out_dims.d[j];
        }

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(this->engine_->getNbBindings() == 2);
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        assert(this->engine_->getBindingDataType(this->inputIndex_) == nvinfer1::DataType::kFLOAT);
        assert(this->engine_->getBindingDataType(this->outputIndex_) == nvinfer1::DataType::kFLOAT);

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

    std::vector<Object> YoloXTensorRT::inference(const cv::Mat& frame)
    {
        // preprocess
        auto pr_img = static_resize(frame);
        float* input_blob = new float[pr_img.total()*3];
        blobFromImage(pr_img, input_blob);

        // inference
        float* output_blob = new float[this->output_size_];
        this->doInference(input_blob, output_blob);

        float scale = std::min(this->input_w_ / (frame.cols*1.0), this->input_h_ / (frame.rows*1.0));

        std::vector<Object> objects;
        decode_outputs(output_blob, this->grid_strides_, objects, this->bbox_conf_thresh_, scale, frame.cols, frame.rows);

        delete input_blob;
        delete output_blob;
        return objects;
    }

    void YoloXTensorRT::doInference(float* input, float* output)
    {
        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        void* buffers[2];

        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers[this->inputIndex_], 3 * this->input_h_ * this->input_w_ * sizeof(float)));
        CHECK(cudaMalloc(&buffers[this->outputIndex_], this->output_size_ * sizeof(float)));

        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[this->inputIndex_], input, 3 * this->input_h_ * this->input_w_ * sizeof(float), cudaMemcpyHostToDevice, stream));
        context_->enqueueV2(buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[this->outputIndex_], this->output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[0]));
        CHECK(cudaFree(buffers[1]));
    }

} // namespace yolox_cpp

