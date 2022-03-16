# YOLOX-ROS-CPP

## Usage

### Requirements
- ROS2 Foxy
- OpenCV 4.x
- OpenVINO *
- TensorRT 8.x *

※ Either one of OpenVINO or TensorRT is required.

※ YOLOX is not required.

※ Jetson + TensorRT docker support.(Jetpack 4.6 r32.6.1)


### Execute with docker

#### OpenVINO
```bash
# base image is `openvino/ubuntu20_dev:2021.4.1_20210416`
docker pull fateshelled/openvino_yolox_ros:latest

xhost +
docker run --rm -it \
           --network host \
           --user openvino \
           -v $HOME/ros2_ws:/home/openvino/ros2_ws \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -w /home/openvino/ros2_ws \
           -e DISPLAY=$DISPLAY \
           --device /dev/video0:/dev/video0 \
           fateshelled/openvino_yolox_ros /bin/bash

```

#### TensorRT
```bash
# base image is `nvcr.io/nvidia/tensorrt:21.09-py3`
docker pull fateshelled/tensorrt_yolox_ros:latest

xhost +
docker run --rm -it \
           --network host \
           --runtime nvidia \
           -v $HOME/ros2_ws:/root/ros2_ws \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -w /root/ros2_ws \
           -e DISPLAY=$DISPLAY \
           --device /dev/video0:/dev/video0 \
           fateshelled/tensorrt_yolox_ros /bin/bash
```

#### Jetson + TensorRT
```bash
# base image is `dustynv/ros:foxy-ros-base-l4t-r32.6.1`
docker pull fateshelled/jetson_yolox_ros:foxy-ros-base-l4t-r32.6.1

# This image cannot display GUI.
docker run --rm -it \
           --network host \
           --runtime nvidia \
           -v $HOME/ros2_ws:/root/ros2_ws \
           -w /root/ros2_ws \
           --device /dev/video0:/dev/video0 \
           fateshelled/jetson_yolox_ros /bin/bash
```


### Clone YOLOX-ROS
```bash
cd ~/ros2_ws/src
git clone https://github.com/fateshelled/YOLOX-ROS -b dev_cpp
```


### Model Convert
#### OpenVINO
```bash
cd ~/ros2_ws

# Download onnx file and Convert to IR format.
./src/YOLOX-ROS/weights/openvino/install.bash yolox_nano
```

#### TensorRT
```bash
cd ~/ros2_ws

# Download onnx model and Convert to TensorRT engine via trtexec.
# 1st arg is model name. 2nd is workspace size. 3rd is trtexec flag(Optional).
./src/YOLOX-ROS/weights/tensorrt/convert.bash yolox_nano 16
```

### build packages
```bash
# # If use openvino
# source /opt/intel/openvino_2021/bin/setupvars.sh

cd ~/ros2_ws
colcon build --symlink-install
source ./install/setup.bash
```

### Run

#### OpenVINO
```bash
# run YOLOX_nano
ros2 launch yolox_ros_cpp yolox_openvino.launch.py

# run other model
ros2 launch yolox_ros_cpp yolox_openvino.launch.py \
    model_path:=install/yolox_ros_cpp/share/yolox_ros_cpp/weights/tensorrt/yolox_s.trt \
    image_size/height:=640 image_size/width:=640

```

#### TensorRT
```bash
# run YOLOX_nano
ros2 launch yolox_ros_cpp yolox_tensorrt.launch.py
```

#### Jetson + TensorRT
Jetson docker container cannot display GUI.
If you want to show image with bounding box drawn, subscribe from host jetson installed ROS2 Eloquent or Dashing.

```bash
# run YOLOX_nano
ros2 launch yolox_ros_cpp yolox_tensorrt_jetson.launch.py
```

### Parameter
#### OpenVINO example
- `model_path`: ./install/yolox_ros_cpp/share/yolox_ros_cpp/weights/openvino/yolox_nano.xml
- `device`: CPU
- `image_size/width`: 416
- `image_size/height`: 416
- `conf`: 0.3
- `nms`: 0.45
- `imshow_isshow`: true
- `src_image_topic_name`: image_raw
- `publish_image_topic_name`: yolox/image_raw
- `publish_boundingbox_topic_name`: yolox/bounding_boxes


#### TensorRT example.
- `model_path`: ./install/yolox_ros_cpp/share/yolox_ros_cpp/weights/tensorrt/yolox_nano.trt
- `device`: "0"
- `image_size/width`: 416
- `image_size/height`: 416
- `conf`: 0.3
- `nms`: 0.45
- `imshow_isshow`: true
- `src_image_topic_name`: image_raw
- `publish_image_topic_name`: yolox/image_raw
- `publish_boundingbox_topic_name`: yolox/bounding_boxes

`device` is GPU id. Must be specified as a `string` type.

### Reference
Reference from YOLOX demo code.
- https://github.com/Megvii-BaseDetection/YOLOX/blob/5183a6716404bae497deb142d2c340a45ffdb175/demo/OpenVINO/cpp/yolox_openvino.cpp
- https://github.com/Megvii-BaseDetection/YOLOX/tree/5183a6716404bae497deb142d2c340a45ffdb175/demo/TensorRT/cpp
