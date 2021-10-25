#include "yolox_cpp/yolox_tensorrt.hpp"

namespace yolox_cpp{

    namespace{
        cv::Mat static_resize(cv::Mat& img, int height, int width) {
            float r = std::min(width / (img.cols*1.0), height / (img.rows*1.0));
            // r = std::min(r, 1.0f);
            int unpad_w = r * img.cols;
            int unpad_h = r * img.rows;
            cv::Mat re(unpad_h, unpad_w, CV_8UC3);
            cv::resize(img, re, re.size());
            cv::Mat out(width, height, CV_8UC3, cv::Scalar(114, 114, 114));
            re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
            return out;
        }

        void generate_grids_and_stride(const int target_size, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
        {
            for (auto stride : strides)
            {
                int num_grid = target_size / stride;
                for (int g1 = 0; g1 < num_grid; g1++)
                {
                    for (int g0 = 0; g0 < num_grid; g0++)
                    {
                        grid_strides.push_back((GridAndStride){g0, g1, stride});
                    }
                }
            }
        }

        inline float intersection_area(const Object& a, const Object& b)
        {
            cv::Rect_<float> inter = a.rect & b.rect;
            return inter.area();
        }

        void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
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

        void qsort_descent_inplace(std::vector<Object>& objects)
        {
            if (objects.empty())
                return;

            qsort_descent_inplace(objects, 0, objects.size() - 1);
        }

        void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
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

    }
    // OpenVINOとは異なる処理
    namespace{
        void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<Object>& objects)
        {
            const int num_class = 80;

            const int num_anchors = grid_strides.size();

            for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
            {
                const int grid0 = grid_strides[anchor_idx].grid0;
                const int grid1 = grid_strides[anchor_idx].grid1;
                const int stride = grid_strides[anchor_idx].stride;

                const int basic_pos = anchor_idx * (num_class + 5);

                // yolox/models/yolo_head.py decode logic
                float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
                float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
                float w = exp(feat_blob[basic_pos+2]) * stride;
                float h = exp(feat_blob[basic_pos+3]) * stride;
                float x0 = x_center - w * 0.5f;
                float y0 = y_center - h * 0.5f;

                float box_objectness = feat_blob[basic_pos+4];
                for (int class_idx = 0; class_idx < num_class; class_idx++)
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

        float* blobFromImage(cv::Mat& img){
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

            float* blob = new float[img.total()*3];
            int channels = 3;
            int img_h = img.rows;
            int img_w = img.cols;
            std::vector<float> mean = {0.485, 0.456, 0.406};
            std::vector<float> std = {0.229, 0.224, 0.225};
            for (size_t c = 0; c < channels; c++) 
            {
                for (size_t  h = 0; h < img_h; h++) 
                {
                    for (size_t w = 0; w < img_w; w++) 
                    {
                        blob[c * img_w * img_h + h * img_w + w] =
                            (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f - mean[c]) / std[c];
                    }
                }
            }
            return blob;
        }

        void decode_outputs(float* prob, std::vector<Object>& objects, 
                            float scale, const int img_w, const int img_h,
                            const int input_w,
                            const float conf_threshold, const float nms_threshold) {
                std::vector<Object> proposals;
                std::vector<int> strides = {8, 16, 32};
                std::vector<GridAndStride> grid_strides;
                generate_grids_and_stride(input_w, strides, grid_strides);
                generate_yolox_proposals(grid_strides, prob,  conf_threshold, proposals);

                qsort_descent_inplace(proposals);

                std::vector<int> picked;
                nms_sorted_bboxes(proposals, picked, nms_threshold);

                int count = picked.size();

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

        void doInference(IExecutionContext& context, float* input, float* output, const int output_size, 
                         cv::Size input_shape, const char* input_blob_name, const char* output_blob_name) {
            const ICudaEngine& engine = context.getEngine();

            // Pointers to input and output device buffers to pass to engine.
            // Engine requires exactly IEngine::getNbBindings() number of buffers.
            void* buffers[2];

            // In order to bind the buffers, we need to know the names of the input and output tensors.
            // Note that indices are guaranteed to be less than IEngine::getNbBindings()
            const int inputIndex = engine.getBindingIndex(input_blob_name);
            const int outputIndex = engine.getBindingIndex(output_blob_name);

            // Create GPU buffers on device
            CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
            CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

            // Create stream
            cudaStream_t stream;
            CHECK(cudaStreamCreate(&stream));

            // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
            CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
            context.enqueue(1, buffers, stream, nullptr);
            CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);

            // Release stream and buffers
            cudaStreamDestroy(stream);
            CHECK(cudaFree(buffers[inputIndex]));
            CHECK(cudaFree(buffers[outputIndex]));
        }

    }

    YoloXTensorRT::YoloXTensorRT(file_name_t path_to_engine, int device,
                                 float nms_th, float conf_th,
                                 int input_width, int input_height,
                                 std::string input_blob_name, std::string output_blob_name)
    :AbsYoloX(nms_th, conf_th, input_width, input_height),
     INPUT_BLOB_NAME_(input_blob_name),
     OUTPUT_BLOB_NAME_(output_blob_name),
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
        int inputIndex = this->engine_->getBindingIndex(this->INPUT_BLOB_NAME_.c_str());
        assert(this->engine_->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
        int outputIndex = this->engine_->getBindingIndex(this->OUTPUT_BLOB_NAME_.c_str());
        assert(this->engine_->getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    }
    YoloXTensorRT::~YoloXTensorRT(){
    }
    std::vector<Object> YoloXTensorRT::inference(cv::Mat frame){
        cv::Mat pr_img = static_resize(frame, this->input_h_, this->input_w_);

        float* blob;
        blob = blobFromImage(pr_img);
        float* prob = new float[this->output_size_];
        doInference(*this->context_, blob, prob, this->output_size_, pr_img.size(), 
                    this->INPUT_BLOB_NAME_.c_str(), this->OUTPUT_BLOB_NAME_.c_str());
        
        float scale = std::min(this->input_w_ / (frame.cols*1.0), this->input_h_ / (frame.rows*1.0));
        
        std::vector<Object> objects;
        decode_outputs(prob, objects, scale, frame.cols, frame.rows, 
                       this->input_w_, this->bbox_conf_thresh_, this->nms_thresh_);

        delete blob;
        return objects;
    }

} // namespace yolox_cpp

