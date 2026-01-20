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

#ifndef _YOLOX_CPP_YOLOX_HAILORT_HPP
#define _YOLOX_CPP_YOLOX_HAILORT_HPP

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "coco_names.hpp"
#include "hailort_common.hpp"
#include "core.hpp"

namespace yolox_cpp
{
  using namespace hailort;
  class YoloXHailoRT: public AbcYoloX
  {
  public:  
    YoloXHailoRT(std::string path_to_model, float nms_th, float conf_th, int num_classes);    
    std::vector<Object> inference(const cv::Mat &src) override;

  protected:
    std::string hef_path_;
    float nms_thresh_;
    float thresh_;
    int num_classes_;

    int output_index_;
    std::vector<std::vector<float>> f_results_;
    std::vector<std::vector<uint8_t>> results_;
    std::vector<int> m_output_strides;
    std::unique_ptr<VDevice> vdevice_;

    std::vector<InputVStream> input_streams_;
    std::vector<OutputVStream> output_streams_;

    std::shared_ptr<hailortCommon::HrtCommon> hrtCommon;
    Expected<std::vector<std::reference_wrapper<Device>>> physical_devices;
    Expected<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vstreams;

    hailo_vstream_info_t input_info_;
  };    
}

#endif // _YOLOX_CPP_YOLOX_HAILORT_HPP