cmake_minimum_required(VERSION 2.8.12)
project(yolox_ros_py)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
    rospy
    sensor_msgs
    cv_bridge
)

set(PATH "${CMAKE_CURRENT_SOURCE_DIR}/../weights")
set(FILE "${PATH}/yolox_s.pth")
message(STATUS "Checking and downloading yolox_s.pth if needed ...")
if (NOT EXISTS "${FILE}")
  message(STATUS "... file does not exist. Downloading now ...")
  execute_process(COMMAND wget -q https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth -P ${PATH})
endif()

set(FILE "${PATH}/yolox_l.pth")
message(STATUS "Checking and downloading yolox_l.pth if needed ...")
if (NOT EXISTS "${FILE}")
  message(STATUS "... file does not exist. Downloading now ...")
  execute_process(COMMAND wget -q https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l.pth -P ${PATH})
endif()

catkin_package()