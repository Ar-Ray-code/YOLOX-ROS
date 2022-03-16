#include "yolox_cpp/yolox_openvino.hpp"

namespace yolox_cpp{
    using namespace InferenceEngine;

    YoloXOpenVINO::YoloXOpenVINO(file_name_t path_to_model, std::string device_name, 
                                 float nms_th, float conf_th, int input_width, int input_height)
    :AbcYoloX(nms_th, conf_th, input_width, input_height)
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
        std::cout << "Loading a model to the device" << std::endl;
        executable_network_ = ie.LoadNetwork(network_, device_name);

        //  Step 5. Create an infer request
        std::cout << "Create an infer request" << std::endl;
        infer_request_ = executable_network_.CreateInferRequest();
            
    }
    std::vector<Object> YoloXOpenVINO::inference(cv::Mat frame){
        // preprocess
        cv::Mat pr_img = static_resize(frame);
        InferenceEngine::Blob::Ptr imgBlob = infer_request_.GetBlob(input_name_);     // just wrap Mat data by Blob::Ptr
        blobFromImage(pr_img, imgBlob);

        // do inference
        /* Running the request synchronously */
        try{
            infer_request_.Infer();
        }catch  (const std::exception& ex) {
            std::cerr << ex.what() << std::endl;
            return {};
        }

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
        
        int img_w = frame.cols;
        int img_h = frame.rows;

        float scale = std::min(input_w_ / (frame.cols*1.0), input_h_ / (frame.rows*1.0));
        
        std::vector<Object> objects;
        decode_outputs(net_pred, objects, scale, img_w, img_h);
        return objects;
    }

    cv::Mat YoloXOpenVINO::static_resize(cv::Mat& img) {
        float r = std::min(input_w_ / (img.cols*1.0), input_h_ / (img.rows*1.0));
        // r = std::min(r, 1.0f);
        int unpad_w = r * img.cols;
        int unpad_h = r * img.rows;
        cv::Mat re(unpad_h, unpad_w, CV_8UC3);
        cv::resize(img, re, re.size());
        //cv::Mat out(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::Mat out(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
        re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
        return out;
    }

    void YoloXOpenVINO::blobFromImage(cv::Mat& img, InferenceEngine::Blob::Ptr& blob){
        int channels = 3;
        int img_h = img.rows;
        int img_w = img.cols;
        InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
        if (!mblob) 
        {
            THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
                << "but by fact we were not able to cast inputBlob to MemoryBlob";
        }
        // locked memory holder should be alive all time while access to its buffer happens
        auto mblobHolder = mblob->wmap();

        float *blob_data = mblobHolder.as<float *>();

        for (size_t c = 0; c < channels; c++) 
        {
            for (size_t  h = 0; h < img_h; h++) 
            {
                for (size_t w = 0; w < img_w; w++) 
                {
                    blob_data[c * img_w * img_h + h * img_w + w] =
                        (float)img.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    }
    void YoloXOpenVINO::generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
    {
        for (auto stride : strides)
        {
            int num_grid_w = target_w / stride;
            int num_grid_h = target_h / stride;
            for (int g1 = 0; g1 < num_grid_h; g1++)
            {
                for (int g0 = 0; g0 < num_grid_w; g0++)
                {
                    grid_strides.push_back((GridAndStride){g0, g1, stride});
                }
            }
        }
    }


    void YoloXOpenVINO::generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr, float prob_threshold, std::vector<Object>& objects)
    {

        const int num_anchors = grid_strides.size();

        for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
        {
            const int grid0 = grid_strides[anchor_idx].grid0;
            const int grid1 = grid_strides[anchor_idx].grid1;
            const int stride = grid_strides[anchor_idx].stride;

            const int basic_pos = anchor_idx * (num_classes_ + 5);

            // yolox/models/yolo_head.py decode logic
            //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
            //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
            float x_center = (feat_ptr[basic_pos + 0] + grid0) * stride;
            float y_center = (feat_ptr[basic_pos + 1] + grid1) * stride;
            float w = exp(feat_ptr[basic_pos + 2]) * stride;
            float h = exp(feat_ptr[basic_pos + 3]) * stride;
            float x0 = x_center - w * 0.5f;
            float y0 = y_center - h * 0.5f;

            float box_objectness = feat_ptr[basic_pos + 4];
            for (int class_idx = 0; class_idx < num_classes_; class_idx++)
            {
                float box_cls_score = feat_ptr[basic_pos + 5 + class_idx];
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

    float YoloXOpenVINO::intersection_area(const Object& a, const Object& b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    void YoloXOpenVINO::qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
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


    void YoloXOpenVINO::qsort_descent_inplace(std::vector<Object>& objects)
    {
        if (objects.empty())
            return;

        qsort_descent_inplace(objects, 0, objects.size() - 1);
    }

    void YoloXOpenVINO::nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
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


    void YoloXOpenVINO::decode_outputs(const float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
            std::vector<Object> proposals;
            std::vector<int> strides = {8, 16, 32};
            std::vector<GridAndStride> grid_strides;

            generate_grids_and_stride(input_w_, input_h_, strides, grid_strides);
            generate_yolox_proposals(grid_strides, prob,  bbox_conf_thresh_, proposals);
            qsort_descent_inplace(proposals);

            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked, nms_thresh_);
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
}