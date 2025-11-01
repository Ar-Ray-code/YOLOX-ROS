cmake_minimum_required(VERSION 3.14)
project(yolox_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")
set(CMAKE_CUDA_ARCHITECTURES 87)
project(yolox_cpp CXX CUDA)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

option(YOLOX_USE_OPENVINO    "Use OpenVINO"  OFF)
option(YOLOX_USE_TENSORRT    "Use TensorRT"  OFF)
option(YOLOX_USE_ONNXRUNTIME "Use ONNXRuntime"  OFF)
option(YOLOX_USE_TFLITE      "Use tflite"  OFF)
option(JETSON                "Use Jetson"  OFF)

if(JETSON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/include/aarch64-linux-gnu/ -L/usr/lib/aarch64-linux-gnu/")
  set(YOLOX_USE_TENSORRT ON)
endif()

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

  set(ENABLE_OPENVINO ON)
  set(TARGET_SRC ${TARGET_SRC} src/yolox_openvino.cpp)
  set(TARGET_LIBS ${TARGET_LIBS} openvino::runtime)
  set(TARGET_DPENDENCIES ${TARGET_DPENDENCIES} OpenVINO)
endif()

if(YOLOX_USE_TENSORRT)
  find_package(CUDA REQUIRED)
  enable_language(CUDA) 
  if (NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
  endif()
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
  set(ENABLE_TENSORRT ON)
  set(TARGET_SRC ${TARGET_SRC} src/yolox_tensorrt.cpp src/cuda_utils.cu)
  set(TARGET_LIBS ${TARGET_LIBS} nvinfer nvinfer_plugin)
  set(TARGET_DPENDENCIES ${TARGET_DPENDENCIES} CUDA)
endif()

# tflite
if(YOLOX_USE_TFLITE)
  set(ENABLE_TFLITE ON)
  set(TARGET_SRC ${TARGET_SRC} src/yolox_tflite.cpp)
  set(TFLITE_INCLUDES
    ${TFLITE_INCLUDE_DIR}
    ${ABSEIL_CPP_ICLUDE_DIR}
    ${FLATBUFFERS_INCLUDE_DIR}
  )
  set(TARGET_LIBS ${TARGET_LIBS} ${TFLITE_LIB_PATH}/libtensorflow-lite.so)
endif()

# onnxruntime
if(YOLOX_USE_ONNXRUNTIME)
  find_library(ONNXRUNTIME NAMES onnxruntime)
  set(ENABLE_ONNXRUNTIME ON)
  set(TARGET_SRC ${TARGET_SRC} src/yolox_onnxruntime.cpp)
  set(TARGET_LIBS ${TARGET_LIBS} onnxruntime)
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/include/yolox_cpp/config.h.in"
  "${PROJECT_SOURCE_DIR}/include/yolox_cpp/config.h"
)

ament_auto_add_library(yolox_cpp SHARED ${TARGET_SRC})
target_include_directories(yolox_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(yolox_cpp ${TARGET_DPENDENCIES})
ament_export_dependencies(${TARGET_DPENDENCIES})
target_link_libraries(yolox_cpp ${TARGET_LIBS})

if (YOLOX_USE_TFLITE)
  target_include_directories(yolox_cpp PUBLIC ${TFLITE_INCLUDES})
  ament_export_include_directories(${TFLITE_INCLUDES})
  install(DIRECTORY ${TFLITE_LIB_PATH}/ DESTINATION lib)
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
