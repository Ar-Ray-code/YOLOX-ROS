// Copyright 2024 Ar-Ray-code
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright 2023 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "yolox_cpp/yolox_hailort.hpp"

namespace yolox_cpp
{
  using namespace hailort;
  YoloXHailoRT::YoloXHailoRT(std::string path_to_model, float nms_th=0.45, float conf_th=0.3, int num_classes=80) :
  AbcYoloX(nms_th, conf_th, "0.1", 8, false), hef_path_(path_to_model),
  nms_thresh_(nms_th), thresh_(conf_th), num_classes_(num_classes)
  {
    m_output_strides = {8, 16, 32};

    hrtCommon = std::make_shared<hailortCommon::HrtCommon>();
    Expected<std::unique_ptr<VDevice>> vdevice = VDevice::create();
    if (!vdevice) std::cerr << "Failed create vdevice, status = " << vdevice.status() << std::endl;
    vdevice_ = std::move(vdevice.value());

    auto network_group = hrtCommon->configureNetworkGroup(*vdevice_, hef_path_);
    if (!network_group) {
      std::cerr << "Failed to configure network group " << hef_path_ << std::endl;
      return;
    }

    Expected<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> temp_vstreams = VStreamsBuilder::create_vstreams(*network_group.value(), QUANTIZED, FORMAT_TYPE);
        if (!temp_vstreams) {
      std::cerr << "Failed creating vstreams " << temp_vstreams.status() << std::endl;
      return;
    }
    input_streams_ = std::move(temp_vstreams->first);
    output_streams_ = std::move(temp_vstreams->second);

    if (input_streams_.size() > MAX_LAYER_EDGES || output_streams_.size() > MAX_LAYER_EDGES) {
      std::cerr << "Trying to infer network with too many input/output virtual streams, Maximum amount is " <<
        MAX_LAYER_EDGES << " (either change HEF or change the definition of MAX_LAYER_EDGES)"<< std::endl;
      return;
    }

    for (output_index_ = 0 ; output_index_ < (int)output_streams_.size(); output_index_++) {
      auto size = output_streams_[output_index_].get_frame_size();
      auto info = output_streams_[output_index_].get_info();
      std::vector<uint8_t> data(size);
      results_.emplace_back(data);      
      std::vector<float> f_data(size*4);
      f_results_.emplace_back(f_data);
      assert(info.format.type == HAILO_FORMAT_TYPE_UINT8);
    }

    input_info_ = input_streams_[0].get_info();
    input_w_ = input_info_.shape.width;
    input_h_ = input_info_.shape.height;
  }

  std::vector<Object> YoloXHailoRT::inference(const cv::Mat &src)
  {
    std::vector<Object> object_array;
    cv::Mat input = this->static_resize(src);

    hrtCommon->infer(input_streams_, output_streams_, input.data, results_);
    std::vector<std::vector<uint8_t>> prev(results_.begin(), results_.end());

    for (int output_index = 0 ; output_index < (int)output_streams_.size(); output_index++) {
      auto info = output_streams_[output_index].get_info();
      auto size = output_streams_[output_index].get_frame_size();

      std::vector<float> data(size);
      std::vector<yolox_cpp::Object> proposals;
      std::vector<int> picked;
      std::vector<GridAndStride> grid_strides;

      hailo_quant_info_t quant_info = info.quant_info;

      for (int i = 0; i < (int)size; i++) {
        int xy = i % info.shape.features;
        int c = i / info.shape.features;
        int index = xy * (4+1+num_classes_) + c;
        f_results_[output_index][index] = hrtCommon->dequant(&prev[output_index][i], quant_info.qp_scale, quant_info.qp_zp, info.format.type);	
      }

      auto input_info = input_streams_[0].get_info();
      const float scale = std::min(input_info.shape.width / (float)src.size().width, input_info.shape.height / (float)src.size().height);

      this->generate_grids_and_stride(input_info.shape.width, input_info.shape.height, m_output_strides, grid_strides);
      this->generate_yolox_proposals(grid_strides, f_results_[output_index].data(), thresh_, proposals);
      if (proposals.size() == 0)
        continue;

      this->qsort_descent_inplace(proposals, 0, proposals.size() - 1);
      this->nms_sorted_bboxes(proposals, picked, nms_thresh_);
      object_array.resize(static_cast<int>(picked.size()));
      this->decode_outputs(f_results_[output_index].data(), grid_strides, object_array, thresh_, scale, src.size().width, src.size().height);
    }

    return object_array;
  }
}

