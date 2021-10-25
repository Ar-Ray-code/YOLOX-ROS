# YOLOX-ROS-CPP

## Usage

### Requirements
- ROS2 Foxy
- OpenVINO *
- TensorRT *
  - Tested in docker container.
    - TensorRT 8.0.3
    - NVIDIA CUDA 11.4.2
    - NVIDIA cuDNN 8.2.4.15

※ YOLOX install is not required.

※ Either one of OpenVINO or TensorRT is required.


### Execute with docker
If don't use docker, ignore this term.

```bash
# OpenVINO image
# base image is `openvino/ubuntu20_dev:2021.4.1_20210416`
docker pull fateshelled/openvino_yolox_ros:latest

xhost +
docker run --rm -it \
           --network host \
           -v $HOME/ros2_ws:/root/ros2_ws \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -w /root/ros2_ws \
           -e DISPLAY=$DISPLAY \
           --device /dev/video0:/dev/video0 \
           fateshelled/openvino_yolox_ros /bin/bash


# TensorRT image
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


### Install YOLOX-ROS-CPP
```bash
cd ~/ros2_ws/src
git clone https://github.com/fateshelled/YOLOX-ROS -b dev_cpp_trt
```

#### build with OpenVINO
```bash
source /opt/ros/foxy/setup.bash
source /opt/intel/openvino_2021/bin/setupvars.sh

cd ~/ros2_ws
colcon build --symlink-install \
             --cmake-args -DYOLOX_USE_TENSORRT=OFF \
                          -DYOLOX_USE_OPENVINO=ON
# Download onnx file and Convert to IR format.
./src/YOLOX-ROS/weights/openvino/install.bash yolox_nano

source ./install/setup.bash
```

#### build with TensorRT
```bash
source /opt/ros/foxy/setup.bash

cd ~/ros2_ws
# Download onnx file and Convert to TensorRT engine via trtexec.
./src/YOLOX-ROS/weights/tensorrt/trtexec/convert yolox_nano

# # If in docker container, you can convert via torch2trt.
# ./src/YOLOX-ROS/weights/tensorrt/convert.bash yolox_nano

colcon build --symlink-install \
             --cmake-args -DYOLOX_USE_TENSORRT=ON \
                          -DYOLOX_USE_OPENVINO=OFF

source ./install/setup.bash
```
※ Input/Output layer name of TRT Engine via `trtexec` is `inputs`/`outputs`, via `torch2trt` is `input_0`/`output_0`.

### DEMO
#### OpenVINO
```bash
# run YOLOX_nano
ros2 launch yolox_ros_cpp yolox_openvino.launch.py
```

#### TensorRT
```bash
# run YOLOX_nano
ros2 launch yolox_ros_cpp yolox_tensorrt.launch.py

# If converted engine with torch2trt, rewrite yolox_tensorrt.launch.py.
# yolox_param_yaml = os.path.join(yolox_ros_share_dir, "param", "nano_trtexec.yaml")
#  ↓
# yolox_param_yaml = os.path.join(yolox_ros_share_dir, "param", "nano_torch2trt.yaml")
```

### Parameter
#### OpenVINO example
- `model_path`: /home/ubuntu/ros2_ws/src/YOLOX-ROS/weights/openvino/yolox_nano.xml
- `model_type`: openvino
- `device`: CPU
- `image_size/width`: 416
- `image_size/height`: 416
- `conf`: 0.3
- `nms`: 0.45
- `imshow_isshow`: true

`input_blob_name` and `output_blob_name` are not use with OpenVINO.

#### TensorRT example. convert via trtexec.
- `model_path`: /home/ubuntu/ros2_ws/src/YOLOX-ROS/weights/tensorrt/trtexec/yolox_nano.trt
- `model_type`: tensorrt
- `device`: "0"
- `image_size/width`: 416
- `image_size/height`: 416
- `input_blob_name`: inputs
- `output_blob_name`: outputs
- `conf`: 0.3
- `nms`: 0.45
- `imshow_isshow`: true

`device` is GPU id. Must be specified as a `string` type.

#### TensorRT example. convert via torch2trt.
- `model_path`: /home/ubuntu/ros2_ws/src/YOLOX-ROS/weights/tensorrt/YOLOX_outputs/yolox_nano.engine
- `model_type`: tensorrt
- `device`: "0"
- `image_size/width`: 416
- `image_size/height`: 416
- `input_blob_name`: input_0
- `output_blob_name`: output_0
- `conf`: 0.3
- `nms`: 0.45
- `imshow_isshow`: true

`device` is GPU id. Must be specified as a `string` type.

### Reference
Reference from YOLOX demo code.
- https://github.com/Megvii-BaseDetection/YOLOX/blob/5183a6716404bae497deb142d2c340a45ffdb175/demo/OpenVINO/cpp/yolox_openvino.cpp
- https://github.com/Megvii-BaseDetection/YOLOX/tree/5183a6716404bae497deb142d2c340a45ffdb175/demo/TensorRT/cpp
