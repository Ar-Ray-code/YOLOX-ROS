cmake_minimum_required(VERSION 3.8)
project(yolox_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

option(YOLOX_USE_OPENVINO    "Use OpenVINO"  OFF)
option(YOLOX_USE_TENSORRT    "Use TensorRT"  OFF)
option(YOLOX_USE_ONNXRUNTIME "Use ONNXRuntime"  OFF)
option(YOLOX_USE_TFLITE      "Use tflite"  OFF)

if(NOT YOLOX_USE_OPENVINO AND NOT YOLOX_USE_TENSORRT AND NOT YOLOX_USE_ONNXRUNTIME AND NOT YOLOX_USE_TFLITE)
  message(FATAL_ERROR "YOLOX_USE_OPENVINO, YOLOX_USE_TENSORRT, YOLOX_USE_ONNXRUNTIME, YOLOX_USE_TFLITE must be ON at least one")
  return()
endif()

set(ENABLE_OPENVINO OFF)
set(ENABLE_TENSORRT OFF)
set(ENABLE_ONNXRUNTIME OFF)
set(ENABLE_TFLITE OFF)

if(YOLOX_USE_OPENVINO)
  find_package(OpenVINO REQUIRED)
  find_package(InferenceEngine REQUIRED)
  find_package(ngraph REQUIRED)

  set(ENABLE_OPENVINO ON)
  set(TARGET_SRC src/yolox_openvino.cpp)
  set(TARGET_LIBS InferenceEngine ngraph)
  set(TARGET_DPENDENCIES OpenVINO InferenceEngine ngraph)
endif()

if(YOLOX_USE_TENSORRT)
  find_package(CUDA REQUIRED)
  find_library(NVINFER  NAMES nvinfer REQUIRED)
  find_library(NVINFERPLUGIN NAMES nvinfer_plugin  REQUIRED)
  find_library(NVPARSERS NAMES nvparsers REQUIRED)
  find_library(NVONNXPARSER NAMES nvonnxparser REQUIRED)

  set(ENABLE_TENSORRT ON)
  set(TARGET_SRC src/yolox_tensorrt.cpp)
  set(TARGET_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)
  set(TARGET_DPENDENCIES CUDA)
endif()


configure_file(
  "${PROJECT_SOURCE_DIR}/include/yolox_cpp/config.h.in"
  "${PROJECT_SOURCE_DIR}/include/yolox_cpp/config.h"
)

ament_auto_add_library(yolox_cpp SHARED ${TARGET_SRC})
ament_target_dependencies(yolox_cpp ${TARGET_DPENDENCIES})
ament_export_dependencies(${TARGET_DPENDENCIES})


if (YOLOX_USE_TENSORRT)
  target_link_libraries(yolox_cpp
    nvinfer
    nvinfer_plugin
    nvparsers
    nvonnxparser
  )
endif()


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
