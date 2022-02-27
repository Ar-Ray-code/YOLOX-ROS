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

# build option
option(YOLOX_USE_OPENVINO  "Use OpenVINO"  ON)
option(YOLOX_USE_TENSORRT  "Use TensorRT"  ON)

set(ENABLE_OPENVINO OFF)
set(ENABLE_TENSORRT OFF)

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
  if( CUDA_FOUND AND NVINFER AND NVINFERPLUGIN AND NVPARSERS AND NVONNXPARSER )
    set(ENABLE_TENSORRT ON)
    set(SRC ${SRC} src/yolox_tensorrt.cpp)
  endif()
endif()

message(STATUS " ENABLE_OPENVINO: ${ENABLE_OPENVINO}")
message(STATUS " ENABLE_TENSORRT: ${ENABLE_TENSORRT}")

if(NOT ENABLE_OPENVINO AND NOT ENABLE_TENSORRT)
  message(WARNING "skip building yolox_cpp, no OpenVINO and TensorRT found")
  return()
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/include/yolox_cpp/config.h.in"
  "${PROJECT_SOURCE_DIR}/include/yolox_cpp/config.h"
)

# Jetson
# include_directories(/usr/local/cuda-10.2/targets/aarch64-linux/include)
# link_directories(/usr/local/cuda-10.2/targets/aarch64-linux/lib)

add_library(yolox_cpp SHARED
  ${SRC}
)

target_compile_definitions(yolox_cpp
  PRIVATE "MY_LIBRARY_BUILDING_LIBRARY"
)
target_compile_options(yolox_cpp PUBLIC -Wall)

target_include_directories(yolox_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
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
