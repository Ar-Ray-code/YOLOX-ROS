#include "yolox_cpp/yolox_tensorrt.hpp"

namespace yolox_cpp{

    YoloXTensorRT::YoloXTensorRT(file_name_t path_to_engine, int device,
                                 float nms_th, float conf_th,
                                 int input_width, int input_height)
    :AbsYoloX(nms_th, conf_th, input_width, input_height),
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
        int input_size = 1;
        for(int j=0;j<input_dims.nbDims;j++) {
            input_size *= input_dims.d[j];
        }
        assert(input_size == 3 * this->input_h_ * this->input_w_);

        auto out_dims = this->engine_->getBindingDimensions(1);
        this->output_size_ = 1;
        for(int j=0;j<out_dims.nbDims;j++) {
            this->output_size_ *= out_dims.d[j];
        }
        
        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(this->engine_->getNbBindings() == 2);
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        assert(this->engine_->getBindingDataType(this->inputIndex_) == nvinfer1::DataType::kFLOAT);
        assert(this->engine_->getBindingDataType(this->outputIndex_) == nvinfer1::DataType::kFLOAT);
    }
    YoloXTensorRT::~YoloXTensorRT(){
    }
    std::vector<Object> YoloXTensorRT::inference(cv::Mat frame){
        this->pr_img_ = static_resize(frame);

        float* input = blobFromImage(this->pr_img_);
        float* output = new float[this->output_size_];
        this->doInference(input, output);
        
        float scale = std::min(this->input_w_ / (frame.cols*1.0), this->input_h_ / (frame.rows*1.0));
        
        std::vector<Object> objects;
        decode_outputs(output, objects, scale, frame.cols, frame.rows);

        delete input;
        delete output;
        return objects;
    }

    cv::Mat YoloXTensorRT::static_resize(cv::Mat& img) {
        float r = std::min(this->input_w_ / (img.cols*1.0), this->input_h_ / (img.rows*1.0));
        // r = std::min(r, 1.0f);
        int unpad_w = r * img.cols;
        int unpad_h = r * img.rows;
        this->re_ = cv::Mat(unpad_h, unpad_w, CV_8UC3);
        cv::resize(img, this->re_, this->re_.size());
        cv::Mat out(this->input_w_, this->input_h_, CV_8UC3, cv::Scalar(114, 114, 114));
        this->re_.copyTo(out(cv::Rect(0, 0, this->re_.cols, this->re_.rows)));
        return out;
    }

    void YoloXTensorRT::generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
    {
        for (auto stride : strides)
        {
            int num_grid = this->input_w_ / stride;
            for (int g1 = 0; g1 < num_grid; g1++)
            {
                for (int g0 = 0; g0 < num_grid; g0++)
                {
                    grid_strides.push_back((GridAndStride){g0, g1, stride});
                }
            }
        }
    }

    float YoloXTensorRT::intersection_area(const Object& a, const Object& b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    void YoloXTensorRT::qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
    {
        int i = left;
        int j = right;
        float p = faceobjects[(left + right) / 2].prob;

        while (i <= j)
        {
            while (faceobjects[i].prob > p)
                i++;

            while (faceobjects[j].prob < p)
                j--;

            if (i <= j)
            {
                // swap
                std::swap(faceobjects[i], faceobjects[j]);

                i++;
                j--;
            }
        }

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                if (left < j) qsort_descent_inplace(faceobjects, left, j);
            }
            #pragma omp section
            {
                if (i < right) qsort_descent_inplace(faceobjects, i, right);
            }
        }
    }

    void YoloXTensorRT::qsort_descent_inplace(std::vector<Object>& objects)
    {
        if (objects.empty())
            return;

        qsort_descent_inplace(objects, 0, objects.size() - 1);
    }

    void YoloXTensorRT::nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
    {
        picked.clear();

        const int n = faceobjects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++)
        {
            areas[i] = faceobjects[i].rect.area();
        }

        for (int i = 0; i < n; i++)
        {
            const Object& a = faceobjects[i];

            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++)
            {
                const Object& b = faceobjects[picked[j]];

                // intersection over union
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                // float IoU = inter_area / union_area
                if (inter_area / union_area > nms_threshold)
                    keep = 0;
            }

            if (keep)
                picked.push_back(i);
        }
    }

    void YoloXTensorRT::doInference(float* input, float* output) {
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
        context_->enqueue(1, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[this->outputIndex_], this->output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[0]));
        CHECK(cudaFree(buffers[1]));
    }

    void YoloXTensorRT::generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<Object>& objects)
    {
        const int num_anchors = grid_strides.size();

        for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
        {
            const int grid0 = grid_strides[anchor_idx].grid0;
            const int grid1 = grid_strides[anchor_idx].grid1;
            const int stride = grid_strides[anchor_idx].stride;

            const int basic_pos = anchor_idx * (this->num_classes_ + 5);

            // yolox/models/yolo_head.py decode logic
            float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
            float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
            float w = exp(feat_blob[basic_pos+2]) * stride;
            float h = exp(feat_blob[basic_pos+3]) * stride;
            float x0 = x_center - w * 0.5f;
            float y0 = y_center - h * 0.5f;

            float box_objectness = feat_blob[basic_pos+4];
            for (int class_idx = 0; class_idx < this->num_classes_; class_idx++)
            {
                float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
                float box_prob = box_objectness * box_cls_score;
                if (box_prob > prob_threshold)
                {
                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = w;
                    obj.rect.height = h;
                    obj.label = class_idx;
                    obj.prob = box_prob;

                    objects.push_back(obj);
                }

            } // class loop

        } // point anchor loop
    }

    float* YoloXTensorRT::blobFromImage(cv::Mat& img){
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        float* blob = new float[img.total()*3];
        const int channels = 3;
        const int img_h = img.rows;
        const int img_w = img.cols;
        for (size_t c = 0; c < channels; c++) 
        {
            for (size_t  h = 0; h < img_h; h++) 
            {
                for (size_t w = 0; w < img_w; w++) 
                {
                    blob[c * img_w * img_h + h * img_w + w] =
                        (float)img.at<cv::Vec3b>(h, w)[c]; // / 255.0f - this->mean_[c]) / this->std_[c];
                }
            }
        }
        return blob;
    }

    void YoloXTensorRT::decode_outputs(float* prob, std::vector<Object>& objects, 
                                       float scale, const int img_w, const int img_h) {
        std::vector<Object> proposals;
        std::vector<int> strides = {8, 16, 32};
        std::vector<GridAndStride> grid_strides;
        generate_grids_and_stride(strides, grid_strides);
        generate_yolox_proposals(grid_strides, prob,  this->bbox_conf_thresh_, proposals);

        qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, this->nms_thresh_);

        int count = picked.size();
        // std::cout << "num of boxes: " << count << std::endl;

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x) / scale;
            float y0 = (objects[i].rect.y) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
    }


} // namespace yolox_cpp

