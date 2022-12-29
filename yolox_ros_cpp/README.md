# YOLOX-ROS-CPP

## Usage

### Requirements
- ROS2 Foxy
- OpenCV 4.x
- OpenVINO 2021.*
- TensorRT 8.x *
- ONNXRuntime *
- Tensorflow Lite *

※ Either one of OpenVINO or TensorRT or ONNXRuntime or Tensorflow Lite is required.

※ ONNXRuntime support CPU or CUDA execute provider.

※ Tensorflow Lite support XNNPACK Delegate only.

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

#### ONNXRuntime
```bash
# base image is "nvcr.io/nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04"
docker pull fateshelled/onnxruntime_yolox_ros:latest

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
           fateshelled/onnxruntime_yolox_ros:latest \
           /bin/bash
```

### Clone YOLOX-ROS
```bash
cd ~/ros2_ws/src
git clone --recursive https://github.com/fateshelled/YOLOX-ROS -b dev_cpp
```


### Model Convert or Download
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

#### ONNXRuntime
```bash
cd ~/ros2_ws
source /opt/ros/foxy/setup.bash

# Download onnx model
./src/YOLOX-ROS/weights/onnx/download.bash yolox_nano
```

#### Tensorflow Lite
```bash
cd ~/ros2_ws

# Download tflite Person Detection model
# https://github.com/Kazuhito00/Person-Detection-using-RaspberryPi-CPU/
./src/YOLOX-ROS/weights/tflite/download_model.bash
```

#### PINTO_model_zoo
- Support PINTO_model_zoo model
- Download model using the following script.
  - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/download_nano.sh
  - `curl -s https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/main/132_YOLOX/download_nano.sh | bash`
  
- ONNX model copy to weight dir
  - `cp resouces_new/saved_model_yolox_nano_480x640/yolox_nano_480x640.onnx ./src/YOLOX-ROS/weights/onnx/`
- Convert to TensorRT engine
  - `./src/YOLOX-ROS/weights/tensorrt/convert.bash yolox_nano_480x640`


### build packages
```bash
# # If use openvino
# source /opt/intel/openvino_2021/bin/setupvars.sh

cd ~/ros2_ws
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
source ./install/setup.bash
```


#### build yolox_ros_cpp with tflite

##### build tflite
https://www.tensorflow.org/lite/guide/build_cmake

Below is an example build script.
Please change `${workspace}` as appropriate for your environment.
```bash
cd ${workspace}
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
mkdir tflite_build
cd tflite_build

cmake ../tensorflow_src/tensorflow/lite \
  -DBUILD_SHARED_LIBS=ON \
  -DTFLITE_ENABLE_INSTALL=OFF \
  -DTFLITE_ENABLE_XNNPACK=ON \
  -DTFLITE_ENABLE_RUY=OFF \
  -DCMAKE_BUILD_TYPE=Release

make -j"$(nproc)"
```

##### build ros package with tflite

This is build script when tflite built as above.

```bash
# build with tflite
colcon build --symlink-install \
  --cmake-args \
    -DYOLOX_USE_TFLITE=ON \
    -DTFLITE_LIB_PATH=${workspace}/tflite_build/libtensorflow-lite.so \
    -DTFLITE_INCLUDE_DIR=${workspace}/tensorflow_src \
    -DABSEIL_CPP_ICLUDE_DIR=${workspace}/tflite_build/abseil-cpp \
    -DFLATBUFFERS_INCLUDE_DIR=${workspace}/tflite_build/flatbuffers/include
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

#### ONNXRuntime
```bash
# run YOLOX_nano
ros2 launch yolox_ros_cpp yolox_onnxruntime.launch.py
```

#### Tensorflow Lite
```bash
# add libtensorflow-lite.so directory path to `LD_LIBRARY_PATH`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${workspace}/tflite_build

# run Person Detection Model
ros2 launch yolox_ros_cpp yolox_tflite.launch.py
```

### Parameter
#### OpenVINO example
- `model_path`: ./install/yolox_ros_cpp/share/yolox_ros_cpp/weights/openvino/yolox_nano.xml
- `class_labels_path`: ""
  - if not set, use coco_names.
  - See [here](https://github.com/fateshelled/YOLOX-ROS/blob/dev_cpp/yolox_ros_cpp/yolox_ros_cpp/labels/coco_names.txt) for label format.
- `num_classes`: 80
- `model_version`: 0.1.1rc0
- `openvino/device`: CPU
- `conf`: 0.3
- `nms`: 0.45
- `imshow_isshow`: true
- `src_image_topic_name`: /image_raw
- `publish_image_topic_name`: /yolox/image_raw
- `publish_boundingbox_topic_name`: /yolox/bounding_boxes


#### TensorRT example.
- `model_path`: ./install/yolox_ros_cpp/share/yolox_ros_cpp/weights/tensorrt/yolox_nano.trt
- `class_labels_path`: ""
- `num_classes`: 80
- `model_version`: 0.1.1rc0
- `tensorrt/device`: 0
- `conf`: 0.3
- `nms`: 0.45
- `imshow_isshow`: true
- `src_image_topic_name`: /image_raw
- `publish_image_topic_name`: /yolox/image_raw
- `publish_boundingbox_topic_name`: /yolox/bounding_boxes


#### ONNXRuntime example.
- `model_path`: ./install/yolox_ros_cpp/share/yolox_ros_cpp/weights/onnx/yolox_nano.onnx
- `class_labels_path`: ""
- `num_classes`: 80
- `model_version`: 0.1.1rc0
- `onnxruntime/use_cuda`: true
- `onnxruntime/use_parallel`: false
- `onnxruntime/device_id`: 0
- `onnxruntime/inter_op_num_threads`: 1
  - if `onnxruntime/use_parallel` is true, the number of threads used to parallelize the execution of the graph (across nodes).
- `onnxruntime/intra_op_num_threads`: 1
  - the number of threads to use to run the model
- `conf`: 0.3
- `nms`: 0.45
- `imshow_isshow`: true
- `src_image_topic_name`: /image_raw
- `publish_image_topic_name`: /yolox/image_raw
- `publish_boundingbox_topic_name`: /yolox/bounding_boxes

#### Tensorflow Lite example.
- `model_path`: ./install/yolox_ros_cpp/share/yolox_ros_cpp/weights/tflite/model.tflite
- `is_nchw`: true
- `class_labels_path`: ""
- `num_classes`: 80
- `model_version`: 0.1.1rc0
- `tflite/num_threads`: 1
- `conf`: 0.3
- `nms`: 0.45
- `imshow_isshow`: true
- `src_image_topic_name`: /image_raw
- `publish_image_topic_name`: /yolox/image_raw
- `publish_boundingbox_topic_name`: /yolox/bounding_boxes


### Reference
Reference from YOLOX demo code.
- https://github.com/Megvii-BaseDetection/YOLOX/blob/5183a6716404bae497deb142d2c340a45ffdb175/demo/OpenVINO/cpp/yolox_openvino.cpp
- https://github.com/Megvii-BaseDetection/YOLOX/tree/5183a6716404bae497deb142d2c340a45ffdb175/demo/TensorRT/cpp
