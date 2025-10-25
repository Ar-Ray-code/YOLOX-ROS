#include "yolox_cpp/yolox_tensorrt.hpp"
#include <vector_types.h>

extern "C" void launchGPUResizeAndBlobFromImage(uchar3* image_data, float* output, int r_width, int r_height,
                                   int input_width, int input_height, float w_ratio, float h_ratio, cudaStream_t stream);

namespace yolox_cpp
{

    YoloXTensorRT::YoloXTensorRT(const file_name_t &path_to_engine, int device,
                                 float nms_th, float conf_th, const std::string &model_version,
                                 int num_classes, bool p6)
        : AbcYoloX(nms_th, conf_th, model_version, num_classes, p6),
          DEVICE_(device)
    {
        cudaSetDevice(this->DEVICE_);
        // create a model using the API directly and serialize it to a stream
        std::vector<char> trtModelStream;
        size_t size{0};

        std::ifstream file(path_to_engine, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }
        else
        {
            std::string msg = "invalid arguments path_to_engine: ";
            msg += path_to_engine;
            throw std::runtime_error(msg.c_str());
        }

        this->runtime_ = std::unique_ptr<IRuntime>(createInferRuntime(this->gLogger_));
        assert(this->runtime_ != nullptr);
        this->engine_ = std::unique_ptr<ICudaEngine>(this->runtime_->deserializeCudaEngine(trtModelStream.data(), size));
        assert(this->engine_ != nullptr);
        this->context_ = std::unique_ptr<IExecutionContext>(this->engine_->createExecutionContext());
        assert(this->context_ != nullptr);

        const auto input_name = this->engine_->getIOTensorName(this->inputIndex_);
        const auto input_dims = this->engine_->getTensorShape(input_name);
        this->input_h_ = input_dims.d[2];
        this->input_w_ = input_dims.d[3];
        std::cout << "INPUT_HEIGHT: " << this->input_h_ << std::endl;
        std::cout << "INPUT_WIDTH: " << this->input_w_ << std::endl;

        const auto output_name = this->engine_->getIOTensorName(this->outputIndex_);
        auto output_dims = this->engine_->getTensorShape(output_name);
        this->output_size_ = 1;
        for (int j = 0; j < output_dims.nbDims; ++j)
        {
            this->output_size_ *= output_dims.d[j];
        }

        // allocate buffer
        this->input_blob_.resize(this->input_h_ * this->input_w_ * 3);
        this->output_blob_.resize(this->output_size_);

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(this->engine_->getNbIOTensors() == 2);
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        assert(this->engine_->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
        assert(this->engine_->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT);

        // Create GPU buffers on device
        CHECK(cudaMalloc(&this->inference_buffers_[this->inputIndex_], 3 * this->input_h_ * this->input_w_ * sizeof(float)));
        CHECK(cudaMalloc(&this->inference_buffers_[this->outputIndex_], this->output_size_ * sizeof(float)));

        assert(this->context_->setInputShape(input_name, input_dims));
        assert(this->context_->allInputDimensionsSpecified());

        assert(this->context_->setInputTensorAddress(input_name, this->inference_buffers_[this->inputIndex_]));
        assert(this->context_->setOutputTensorAddress(output_name, this->inference_buffers_[this->outputIndex_]));

        // Prepare GridAndStrides
        if (this->p6_)
        {
            generate_grids_and_stride(this->input_w_, this->input_h_, this->strides_p6_, this->grid_strides_);
        }
        else
        {
            generate_grids_and_stride(this->input_w_, this->input_h_, this->strides_, this->grid_strides_);
        }
    }

    YoloXTensorRT::~YoloXTensorRT()
    {
        CHECK(cudaFree(inference_buffers_[this->inputIndex_]));
        CHECK(cudaFree(inference_buffers_[this->outputIndex_]));
    }

    std::vector<yolox_cpp::Object> YoloXTensorRT::inference(const cv::Mat &frame, uchar3* d_image, float* d_output, 
                                                            cudaStream_t copy_stream_, cudaStream_t resize_stream_)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        cudaMemcpy2DAsync(
            d_image,                       
            frame.cols * sizeof(uchar3),  
            frame.data,                 
            frame.step,                    
            frame.cols * sizeof(uchar3),   
            frame.rows,                   
            cudaMemcpyHostToDevice,
            copy_stream_
        );

        auto t1 = std::chrono::high_resolution_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        printf("Copy Time: %5ld us\n", elapsed_us);

        float w_r = static_cast<float>(frame.cols) / this->input_w_;
        float h_r = static_cast<float>(frame.rows) / this->input_h_;

        t0 = std::chrono::high_resolution_clock::now();
        launchGPUResizeAndBlobFromImage(d_image, d_output,
                                        this->input_w_, this->input_h_,
                                        frame.cols, frame.rows,
                                        w_r, h_r,
                                        resize_stream_);
        cudaStreamSynchronize(resize_stream_);
        t1 = std::chrono::high_resolution_clock::now();
        elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        printf("resize time on GPU: %5ld us\n", elapsed_us);

        t0 = std::chrono::high_resolution_clock::now();
        this->doInference(d_output, output_blob_.data(), copy_stream_);
        t1 = std::chrono::high_resolution_clock::now();
        elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        printf("Inference time on GPU: %5ld us\n", elapsed_us);

        const float scale = std::min(
            static_cast<float>(this->input_w_) / frame.cols,
            static_cast<float>(this->input_h_) / frame.rows
        );

        std::vector<Object> objects;
        t0 = std::chrono::high_resolution_clock::now();
        decode_outputs(output_blob_.data(), this->grid_strides_, objects,
                    this->bbox_conf_thresh_, scale, frame.cols, frame.rows);
        t1 = std::chrono::high_resolution_clock::now();
        elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        printf("Post time: %5ld us\n", elapsed_us);

        return objects;
    }


    void YoloXTensorRT::doInference(const float *input, float *output, cudaStream_t copy_stream_)
    {
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        // Change to Device to Device copy
        this->inference_buffers_[this->inputIndex_] = const_cast<float*>(input);

        bool success = context_->executeV2(this->inference_buffers_);
        if (!success)
            throw std::runtime_error("failed inference");

        CHECK(
            cudaMemcpyAsync(
                output,
                this->inference_buffers_[this->outputIndex_],
                this->output_size_ * sizeof(float),
                cudaMemcpyDeviceToHost, copy_stream_));

        CHECK(cudaStreamSynchronize(copy_stream_));
    }

} // namespace yolox_cpp
