cmake_minimum_required(VERSION 3.14)
project(yolox_ros_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")
set(CMAKE_CUDA_ARCHITECTURES 87)
project(yolox_ros_cpp CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(tr_messages REQUIRED)


ament_auto_find_build_dependencies()

if(NOT yolox_cpp_FOUND)
  message(WARNING "skipping yolox_ros_cpp, no yolox_cpp found")
  return()
endif()

ament_auto_add_library(yolox_ros_cpp SHARED
  src/yolox_ros_cpp.cpp
)
rclcpp_components_register_node(
  yolox_ros_cpp
  PLUGIN "yolox_ros_cpp::YoloXNode"
  EXECUTABLE yolox_ros_cpp_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  launch
)

ament_target_dependencies(yolox_ros_cpp tr_messages)
