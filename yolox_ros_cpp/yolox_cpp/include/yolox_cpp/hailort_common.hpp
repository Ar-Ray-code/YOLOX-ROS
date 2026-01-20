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

#ifndef _HAILORT_COMMON
#define _HAILORT_COMMON

#include <algorithm>
#include <iostream>
#include <hailo/hailort.hpp>
#include <sstream>
#include <thread>

using namespace hailort;

#define SAMPLING_PERIOD (HAILO_SAMPLING_PERIOD_332US)
#define AVERAGE_FACTOR (HAILO_AVERAGE_FACTOR_16)
#define DVM_OPTION (HAILO_DVM_OPTIONS_AUTO) // For current measurement over EVB - pass DVM explicitly (see hailo_dvm_options_t)
#define MEASUREMENT_BUFFER_INDEX (HAILO_MEASUREMENT_BUFFER_INDEX_0)

constexpr size_t FRAMES_COUNT = 1;
constexpr bool QUANTIZED = true;
constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;
constexpr size_t MAX_LAYER_EDGES = 16;

extern void writeAll(InputVStream &input, hailo_status &status, unsigned char *preprocessed);
extern void readAll(OutputVStream &output, hailo_status &status,  std::vector<uint8_t> &data);

namespace hailortCommon
{
  template <typename T>
  float dequantInt(T qv, const float qp_scale, const float qp_zp) {
  return ((float)qv - qp_zp) * qp_scale;
  }

  void writeAll(InputVStream &input, hailo_status &status, unsigned char *preprocessed)
  {
    for (size_t i = 0; i < FRAMES_COUNT; i++) {
      status = input.write(MemoryView(preprocessed, input.get_frame_size()));      
      if (HAILO_SUCCESS != status) {
      	return;
      }
    }

    // Flushing is not mandatory here
    status = input.flush();
    if (HAILO_SUCCESS != status) {
      std::cerr << "Failed flushing input vstream" << std::endl;
      return;
    }

    status = HAILO_SUCCESS;
    return;
  }

  void readAll(OutputVStream &output, hailo_status &status,  std::vector<uint8_t> &data)
  {
    for (size_t i = 0; i < FRAMES_COUNT; i++) {
      status = output.read(MemoryView(data.data(), data.size()));
      if (HAILO_SUCCESS != status) {
	return;
      }
    }
    status = HAILO_SUCCESS;
    return;
  }

  class HrtCommon
  {
  public:
    HrtCommon() = default;
    ~HrtCommon() = default;

    float dequant(void *qv,  const float qp_scale, const float qp_zp, hailo_format_type_t format)
    {
      float dqv=0.0;
      if (format == HAILO_FORMAT_TYPE_UINT16) {
        dqv = dequantInt(*(uint16_t *)qv, qp_scale, qp_zp);
      } else if (format == HAILO_FORMAT_TYPE_UINT8) {
        dqv = dequantInt(*(uint8_t *)qv, qp_scale, qp_zp);
      } else if (format == HAILO_FORMAT_TYPE_FLOAT32) {
        dqv = *(float *)qv;
      } else {
        std::cout << "Warn : Unsupport Format" << std::endl;
      }
      return dqv;
    }

    Expected<std::shared_ptr<ConfiguredNetworkGroup>>
    configureNetworkGroup(VDevice &vdevice, std::string &hef_file)
    {
      auto hef = Hef::create(hef_file);
      if (!hef) {
        return make_unexpected(hef.status());
      }

      auto configure_params = vdevice.create_configure_params(hef.value());
      if (!configure_params) {
        return make_unexpected(configure_params.status());
      }

      auto network_groups = vdevice.configure(hef.value(), configure_params.value());
      if (!network_groups) {
        return make_unexpected(network_groups.status());
      }

      if (1 != network_groups->size()) {
        std::cerr << "Invalid amount of network groups" << std::endl;
        return make_unexpected(HAILO_INTERNAL_FAILURE);
      }
      return std::move(network_groups->at(0));
    }

    hailo_status infer(std::vector<InputVStream> &input_streams, std::vector<OutputVStream> &output_streams, unsigned char *data, std::vector<std::vector<uint8_t>> &results)
    {
        hailo_status status = HAILO_SUCCESS; // Success oriented
        hailo_status input_status[MAX_LAYER_EDGES] = {HAILO_UNINITIALIZED};
        hailo_status output_status[MAX_LAYER_EDGES] = {HAILO_UNINITIALIZED};
        std::unique_ptr<std::thread> input_threads[MAX_LAYER_EDGES];
        std::unique_ptr<std::thread> output_threads[MAX_LAYER_EDGES];
        size_t input_thread_index = 0;
        size_t output_thread_index = 0;
        // Create read threads

        for (output_thread_index = 0 ; output_thread_index < output_streams.size(); output_thread_index++) {
        output_threads[output_thread_index] = std::make_unique<std::thread>(readAll,
                                        std::ref(output_streams[output_thread_index]), std::ref(output_status[output_thread_index]), std::ref(results[output_thread_index]));
        }

        // Create write threads
        for (input_thread_index = 0 ; input_thread_index < input_streams.size(); input_thread_index++) {
        input_threads[input_thread_index] = std::make_unique<std::thread>(writeAll,
                                        std::ref(input_streams[input_thread_index]), std::ref(input_status[input_thread_index]), data);
        }

        // Join write threads
        for (size_t i = 0; i < input_thread_index; i++) {
        input_threads[i]->join();
        if (HAILO_SUCCESS != input_status[i]) {
            status = input_status[i];
        }
        }

        // Join read threads
        for (size_t i = 0; i < output_thread_index; i++) {
        output_threads[i]->join();
        if (HAILO_SUCCESS != output_status[i]) {
            status = output_status[i];
        }
        }

        return status;
    }
  };
}

#endif // _HAILORT_COMMON