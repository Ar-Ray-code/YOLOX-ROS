# YOLOX-ROS-CPP

## Usage

### Requirements
- ROS2 Foxy
- OpenCV 4.x
- OpenVINO 2021.*
- TensorRT 8.x *

※ Either one of OpenVINO or TensorRT is required.

※ Model convert script is not supported OpenVINO 2022.*

※ YOLOX is not required.

※ Jetson + TensorRT docker support (Jetpack 4.6 r32.6.1). Tested with Jetson Nano 4GB.


### Execute with docker

#### OpenVINO
```bash
# base image is "openvino/ubuntu20_dev:2021.4.1_20210416"
docker pull fateshelled/openvino_yolox_ros:latest

xhost +
docker run --rm -it \
           --network host \
           --privileged \
           --user openvino \
           -v $HOME/ros2_ws:/home/openvino/ros2_ws \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -w /home/openvino/ros2_ws \
           -e DISPLAY=$DISPLAY \
           --device /dev/video0:/dev/video0 \
           fateshelled/openvino_yolox_ros:latest /bin/bash

# If use NCS2, mount "/dev/bus/usb".
xhost +
docker run --rm -it \
           --network host \
           --privileged \
           --user openvino \
           -v $HOME/ros2_ws:/home/openvino/ros2_ws \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -w /home/openvino/ros2_ws \
           -v /dev/bus/usb:/dev/bus/usb
           -e DISPLAY=$DISPLAY \
           --device /dev/video0:/dev/video0 \
           fateshelled/openvino_yolox_ros:latest \
           /bin/bash

```

#### TensorRT
```bash
# base image is "nvcr.io/nvidia/pytorch:21.09-py3"
docker pull swiftfile/tensorrt_yolox_ros:latest

xhost +
docker run --rm -it \
           --network host \
           --gpus all \
           --privileged \
           -v $HOME/ros2_ws:/root/ros2_ws \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -w /root/ros2_ws \
           -e DISPLAY=$DISPLAY \
           --device /dev/video0:/dev/video0 \
           swiftfile/tensorrt_yolox_ros:latest \
           /bin/bash
```

#### Jetson + TensorRT
```bash
# base image is "dustynv/ros:foxy-ros-base-l4t-r32.6.1"
docker pull fateshelled/jetson_yolox_ros:foxy-ros-base-l4t-r32.6.1

# This image cannot display GUI.
docker run --rm -it \
           --network host \
           --runtime nvidia \
           -v $HOME/ros2_ws:/root/ros2_ws \
           -w /root/ros2_ws \
           --device /dev/video0:/dev/video0 \
           fateshelled/jetson_yolox_ros:foxy-ros-base-l4t-r32.6.1 \
           /bin/bash
```


### Clone YOLOX-ROS
```bash
cd ~/ros2_ws/src
git clone --recursive https://github.com/fateshelled/YOLOX-ROS -b dev_cpp
```


### Model Convert
#### OpenVINO
```bash
cd ~/ros2_ws

# Download onnx file and convert to IR format.
./src/YOLOX-ROS/weights/openvino/install.bash yolox_nano
```

#### TensorRT
```bash
cd ~/ros2_ws

# Download onnx model and convert to TensorRT engine.
# 1st arg is model name. 2nd is workspace size.
./src/YOLOX-ROS/weights/tensorrt/convert.bash yolox_nano 16
```

#### PINTO_model_zoo
- Support PINTO_model_zoo model
- Download model using the following script.
  - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/download_nano.sh
- ONNX model copy to weight dir
  - `cp saved_model_yolox_nano_480x640/yolox_nano_480x640.onnx src/YOLOX-ROS/weight/onnx/`
- Convert to TensorRT engine
  - `./src/YOLOX-ROS/weights/tensorrt/convert.bash yolox_nano_480x640`


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
    model_path:=install/yolox_ros_cpp/share/yolox_ros_cpp/weights/openvino/yolox_s.xml

# run PINTO_model_zoo model
# This model is converted from version 0.1.0.
ros2 launch yolox_ros_cpp yolox_openvino.launch.py \
    model_path:=install/yolox_ros_cpp/share/yolox_ros_cpp/weights/onnx/yolox_nano_480x640.onnx \
    model_version:="0.1.0"

# run YOLOX-tiny with NCS2
ros2 launch yolox_ros_cpp yolox_openvino_ncs2.launch.py

```

#### TensorRT
```bash
# run YOLOX_nano
ros2 launch yolox_ros_cpp yolox_tensorrt.launch.py

# run PINTO_model_zoo model
# This model is converted from version 0.1.0.
ros2 launch yolox_ros_cpp yolox_tensorrt.launch.py \
    model_path:=install/yolox_ros_cpp/share/yolox_ros_cpp/weights/tensorrt/yolox_nano_480x640.trt \
    model_version:="0.1.0"

```

#### Jetson + TensorRT
Jetson docker container cannot display GUI.
If you want to show image with bounding box drawn, subscribe from host jetson or other PC.

```bash
# run YOLOX_nano
ros2 launch yolox_ros_cpp yolox_tensorrt_jetson.launch.py
```

### Parameter
#### OpenVINO example
- `model_path`: ./install/yolox_ros_cpp/share/yolox_ros_cpp/weights/openvino/yolox_nano.xml
- `model_version`: 0.1.1rc0
- `device`: CPU
- `conf`: 0.3
- `nms`: 0.45
- `imshow_isshow`: true
- `src_image_topic_name`: image_raw
- `publish_image_topic_name`: yolox/image_raw
- `publish_boundingbox_topic_name`: yolox/bounding_boxes


#### TensorRT example.
- `model_path`: ./install/yolox_ros_cpp/share/yolox_ros_cpp/weights/tensorrt/yolox_nano.trt
- `model_version`: 0.1.1rc0
- `device`: "0"
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
