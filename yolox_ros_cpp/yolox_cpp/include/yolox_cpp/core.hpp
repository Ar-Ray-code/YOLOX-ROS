#ifndef _YOLOX_CPP_CORE_HPP
#define _YOLOX_CPP_CORE_HPP

#include <opencv2/core/types.hpp>

namespace yolox_cpp
{
/**
 * @brief Define names based depends on Unicode path support
 */
#define tcout std::cout
#define file_name_t std::string
#define imread_t cv::imread

    struct Object
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
    };

    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
    };

    class AbcYoloX
    {
    public:
        AbcYoloX() {}
        AbcYoloX(float nms_th = 0.45, float conf_th = 0.3,
                 std::string model_version = "0.1.1rc0",
                 int num_classes = 80, bool p6 = false)
            : nms_thresh_(nms_th), bbox_conf_thresh_(conf_th),
              model_version_(model_version), num_classes_(num_classes), p6_(p6)
        {
        }
        virtual std::vector<Object> inference(const cv::Mat &frame) = 0;

    protected:
        int input_w_;
        int input_h_;
        float nms_thresh_;
        float bbox_conf_thresh_;
        int num_classes_;
        bool p6_;
        std::string model_version_;
        const std::vector<float> mean_ = {0.485, 0.456, 0.406};
        const std::vector<float> std_ = {0.229, 0.224, 0.225};
        const std::vector<int> strides_ = {8, 16, 32};
        const std::vector<int> strides_p6_ = {8, 16, 32, 64};
        std::vector<GridAndStride> grid_strides_;

        cv::Mat static_resize(const cv::Mat &img)
        {
            float r = std::min(input_w_ / (img.cols * 1.0), input_h_ / (img.rows * 1.0));
            // r = std::min(r, 1.0f);
            int unpad_w = r * img.cols;
            int unpad_h = r * img.rows;
            cv::Mat re(unpad_h, unpad_w, CV_8UC3);
            cv::resize(img, re, re.size());
            cv::Mat out(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
            re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
            return out;
        }

        // for NCHW
        void blobFromImage(const cv::Mat &img, float *blob_data)
        {
            size_t channels = 3;
            size_t img_h = img.rows;
            size_t img_w = img.cols;
            if (this->model_version_ == "0.1.0")
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t h = 0; h < img_h; ++h)
                    {
                        for (size_t w = 0; w < img_w; ++w)
                        {
                            blob_data[c * img_w * img_h + h * img_w + w] =
                                ((float)img.ptr<cv::Vec3b>(h)[w][c] / 255.0 - this->mean_[c]) / this->std_[c];
                        }
                    }
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t h = 0; h < img_h; ++h)
                    {
                        for (size_t w = 0; w < img_w; ++w)
                        {
                            blob_data[c * img_w * img_h + h * img_w + w] = (float)img.ptr<cv::Vec3b>(h)[w][c]; // 0.1.1rc0 or later
                        }
                    }
                }
            }
        }

        // for NHWC
        void blobFromImage_nhwc(const cv::Mat &img, float *blob_data)
        {
            size_t channels = 3;
            size_t img_h = img.rows;
            size_t img_w = img.cols;
            if (this->model_version_ == "0.1.0")
            {
                for (size_t i = 0; i < img_h * img_w; ++i)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        blob_data[i * channels + c] =
                            ((float)img.data[i * channels + c] / 255.0 - this->mean_[c]) / this->std_[c];
                    }
                }
            }
            else
            {
                for (size_t i = 0; i < img_h * img_w; ++i)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        blob_data[i * channels + c] = (float)img.data[i * channels + c]; // 0.1.1rc0 or later
                    }
                }
            }
        }

        void generate_grids_and_stride(const int target_w, const int target_h, const std::vector<int> &strides, std::vector<GridAndStride> &grid_strides)
        {
            for (auto stride : strides)
            {
                int num_grid_w = target_w / stride;
                int num_grid_h = target_h / stride;
                for (int g1 = 0; g1 < num_grid_h; ++g1)
                {
                    for (int g0 = 0; g0 < num_grid_w; ++g0)
                    {
                        grid_strides.push_back((GridAndStride){g0, g1, stride});
                    }
                }
            }
        }

        void generate_yolox_proposals(const std::vector<GridAndStride> grid_strides, const float *feat_ptr, const float prob_threshold, std::vector<Object> &objects)
        {
            const int num_anchors = grid_strides.size();

            for (int anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx)
            {
                const int grid0 = grid_strides[anchor_idx].grid0;
                const int grid1 = grid_strides[anchor_idx].grid1;
                const int stride = grid_strides[anchor_idx].stride;

                const int basic_pos = anchor_idx * (num_classes_ + 5);

                float box_objectness = feat_ptr[basic_pos + 4];
                int class_id = 0;
                float max_class_score = 0.0;
                for (int class_idx = 0; class_idx < num_classes_; ++class_idx)
                {
                    float box_cls_score = feat_ptr[basic_pos + 5 + class_idx];
                    float box_prob = box_objectness * box_cls_score;
                    if (box_prob > max_class_score)
                    {
                        class_id = class_idx;
                        max_class_score = box_prob;
                    }
                }
                if (max_class_score > prob_threshold)
                {
                    // yolox/models/yolo_head.py decode logic
                    //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
                    //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
                    float x_center = (feat_ptr[basic_pos + 0] + grid0) * stride;
                    float y_center = (feat_ptr[basic_pos + 1] + grid1) * stride;
                    float w = exp(feat_ptr[basic_pos + 2]) * stride;
                    float h = exp(feat_ptr[basic_pos + 3]) * stride;
                    float x0 = x_center - w * 0.5f;
                    float y0 = y_center - h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = w;
                    obj.rect.height = h;
                    obj.label = class_id;
                    obj.prob = max_class_score;
                    objects.push_back(obj);
                }
            } // point anchor loop
        }

        float intersection_area(const Object &a, const Object &b)
        {
            cv::Rect_<float> inter = a.rect & b.rect;
            return inter.area();
        }

        void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right)
        {
            int i = left;
            int j = right;
            float p = faceobjects[(left + right) / 2].prob;

            while (i <= j)
            {
                while (faceobjects[i].prob > p)
                    ++i;

                while (faceobjects[j].prob < p)
                    --j;

                if (i <= j)
                {
                    std::swap(faceobjects[i], faceobjects[j]);

                    ++i;
                    --j;
                }
            }
            if (left < j)
                qsort_descent_inplace(faceobjects, left, j);
            if (i < right)
                qsort_descent_inplace(faceobjects, i, right);
        }

        void qsort_descent_inplace(std::vector<Object> &objects)
        {
            if (objects.empty())
                return;

            qsort_descent_inplace(objects, 0, objects.size() - 1);
        }

        void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, const float nms_threshold)
        {
            picked.clear();

            const int n = faceobjects.size();

            std::vector<float> areas(n);
            for (int i = 0; i < n; ++i)
            {
                areas[i] = faceobjects[i].rect.area();
            }

            for (int i = 0; i < n; ++i)
            {
                const Object &a = faceobjects[i];
                const int picked_size = picked.size();

                int keep = 1;
                for (int j = 0; j < picked_size; ++j)
                {
                    const Object &b = faceobjects[picked[j]];

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

        void decode_outputs(const float *prob, const std::vector<GridAndStride> grid_strides,
                            std::vector<Object> &objects, const float bbox_conf_thresh,
                            const float scale, const int img_w, const int img_h)
        {

            std::vector<Object> proposals;
            generate_yolox_proposals(grid_strides, prob, bbox_conf_thresh, proposals);

            qsort_descent_inplace(proposals);

            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked, nms_thresh_);

            int count = picked.size();
            objects.resize(count);

            for (int i = 0; i < count; ++i)
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
    };
}
#endif