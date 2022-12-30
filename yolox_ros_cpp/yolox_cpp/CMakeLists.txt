cmake_minimum_required(VERSION 3.5)
project(yolox_cpp)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# build option
option(YOLOX_USE_OPENVINO    "Use OpenVINO"  ON)
option(YOLOX_USE_TENSORRT    "Use TensorRT"  ON)
option(YOLOX_USE_ONNXRUNTIME "Use ONNXRuntime"  ON)
option(YOLOX_USE_TFLITE      "Use tflite"  OFF)
set(TFLITE_LIB_PATH "" CACHE PATH "Path to libtensorflow-lite.so")
set(TFLITE_INCLUDE_DIR "" CACHE PATH "Header directory of tflite")
set(ABSEIL_CPP_ICLUDE_DIR "" CACHE PATH "Header directory of abseil-cpp")
set(FLATBUFFERS_INCLUDE_DIR "" CACHE PATH "Header directory of flatbuffers")

set(ENABLE_OPENVINO OFF)
set(ENABLE_TENSORRT OFF)
set(ENABLE_ONNXRUNTIME OFF)
set(ENABLE_TFLITE OFF)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)

if(YOLOX_USE_OPENVINO)
  find_package(InferenceEngine)
  find_package(ngraph)
  if( InferenceEngine_FOUND AND ngraph_FOUND )
    set(ENABLE_OPENVINO ON)
    set(SRC ${SRC} src/yolox_openvino.cpp)
  endif()
endif()
if(YOLOX_USE_TENSORRT)
  find_package(CUDA)
  find_library(NVINFER  NAMES nvinfer)
  find_library(NVINFERPLUGIN NAMES nvinfer_plugin)
  find_library(NVPARSERS NAMES nvparsers)
  find_library(NVONNXPARSER NAMES nvonnxparser)
  find_library(NVONNXPARSERRUNTIME NAMES nvonnxparser_runtime)
  if(NOT CUDA_FOUND)
    message(WARNING " CUDA not found")
  endif()
  if(NOT NVINFER)
    message(WARNING " NVINFER not found")
  endif()
  if(NOT NVINFERPLUGIN)
    message(WARNING " NVINFERPLUGIN not found")
  endif()
  if(NOT NVPARSERS)
    message(WARNING " NVPARSERS not found")
  endif()
  if(NOT NVONNXPARSER)
    message(WARNING " NVONNXPARSER not found")
  endif()
  # message(WARNING " NVONNXPARSERRUNTIME ${NVONNXPARSERRUNTIME}") #not use
  if( CUDA_FOUND AND NVINFER AND NVINFERPLUGIN AND NVPARSERS AND NVONNXPARSER )
    message(STATUS " CUDA ${CUDA_FOUND}")
    message(STATUS " NVINFER ${NVINFER}")
    message(STATUS " NVINFERPLUGIN ${NVINFERPLUGIN}")
    message(STATUS " NVPARSERS ${NVPARSERS}")
    message(STATUS " NVONNXPARSER ${NVONNXPARSER}")
    set(ENABLE_TENSORRT ON)
    set(SRC ${SRC} src/yolox_tensorrt.cpp)
  endif()
endif()
if(YOLOX_USE_ONNXRUNTIME)
  find_library(ONNXRUNTIME NAMES onnxruntime)
  if(NOT ONNXRUNTIME)
    message(WARNING " ONNXRUNTIME not found")
  else()
    message(STATUS " ONNXRUNTIME ${ONNXRUNTIME}")
    set(ENABLE_ONNXRUNTIME ON)
    set(SRC ${SRC} src/yolox_onnxruntime.cpp)
  endif()
endif()
if(YOLOX_USE_TFLITE)
  if(NOT ${TFLITE_LIB_PATH})
    set(ENABLE_TFLITE ON)
    set(SRC ${SRC} src/yolox_tflite.cpp)
    set(INCLUDES ${INCLUDES} ${TFLITE_INCLUDE_DIR})
    set(INCLUDES ${INCLUDES} ${ABSEIL_CPP_ICLUDE_DIR})
    set(INCLUDES ${INCLUDES} ${FLATBUFFERS_INCLUDE_DIR})
  else()
    message(WARNING "TFLITE_LIB_PATH is not set")
  endif()
endif()

message(STATUS " ENABLE_OPENVINO: ${ENABLE_OPENVINO}")
message(STATUS " ENABLE_TENSORRT: ${ENABLE_TENSORRT}")
message(STATUS " ENABLE_ONNXRUNTIME: ${ENABLE_ONNXRUNTIME}")
message(STATUS " ENABLE_TFLITE: ${ENABLE_TFLITE}")

if(NOT ENABLE_OPENVINO AND NOT ENABLE_TENSORRT AND NOT ENABLE_ONNXRUNTIME AND NOT ENABLE_TFLITE)
  message(WARNING "skip building yolox_cpp, no OpenVINO, TensorRT, ONNXRuntime and tflite found")
  return()
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/include/yolox_cpp/config.h.in"
  "${PROJECT_SOURCE_DIR}/include/yolox_cpp/config.h"
)

set(INCLUDES ${INCLUDES} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
set(INCLUDES ${INCLUDES} $<INSTALL_INTERFACE:include>)

add_library(yolox_cpp SHARED
  ${SRC}
)

target_compile_definitions(yolox_cpp
  PRIVATE "MY_LIBRARY_BUILDING_LIBRARY"
)
target_compile_options(yolox_cpp PUBLIC -Wall)

target_include_directories(yolox_cpp PUBLIC
  ${INCLUDES}
)

ament_target_dependencies(yolox_cpp
  OpenCV
)
if(ENABLE_OPENVINO)
  ament_target_dependencies(yolox_cpp
    InferenceEngine
    ngraph
  )
endif()
if(ENABLE_TENSORRT)
  target_link_libraries(yolox_cpp
    nvinfer
    nvinfer_plugin
    nvparsers
    nvonnxparser
    # nvonnxparser_runtime
  )
  ament_target_dependencies(yolox_cpp
    CUDA
  )
endif()
if(ENABLE_ONNXRUNTIME)
  target_link_libraries(yolox_cpp
    onnxruntime
  )
endif()
if(ENABLE_TFLITE)
  target_link_libraries(yolox_cpp
    ${TFLITE_LIB_PATH}
  )
endif()



if(NOT WIN32)
  ament_environment_hooks(
    "${ament_cmake_package_templates_ENVIRONMENT_HOOK_LIBRARY_PATH}"
  )
endif()
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  OpenCV
)
if(ENABLE_OPENVINO)
  ament_export_dependencies(
    InferenceEngine
    ngraph
  )
endif()
if(ENABLE_TENSORRT)
  ament_export_dependencies(
    CUDA
  )
endif()

install(TARGETS yolox_cpp
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY
  DESTINATION share/${PROJECT_NAME}
)
install(
  DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
