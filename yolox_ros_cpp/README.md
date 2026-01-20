# YOLOX-ROS-CPP

## Requirements
- ROS2 Iron
  - ros-jazzy-generate-parameter-library
  - ros-jazzy-vision-msgs
  - ros-jazzy-usb-cam
- OpenCV 4.x
- OpenVINO 2024.*
- TensorRT 10.x *
- ONNXRuntime *
- Tensorflow Lite *
- **CUDA 12**

※ Either one of OpenVINO or TensorRT or ONNXRuntime or Tensorflow Lite is required.

<!-- ※ ONNXRuntime support CPU or CUDA execute provider. -->

※ Tensorflow Lite support XNNPACK Delegate only.

※ Tensorflow Lite support float model and does not support integer model.

## Clone YOLOX-ROS
```bash
cd ~/ros2_ws/src
git clone --recursive https://github.com/Ar-Ray-code/YOLOX-ROS -b jazzy
```

## Model Convert or Download
### OpenVINO・ONNXRuntime
```bash
cd ~/ros2_ws

./src/YOLOX-ROS/weights/onnx/download.bash yolox_tiny
```

### TensorRT
```bash
cd ~/ros2_ws

# Download onnx model and convert to TensorRT engine.
# argument is model name. set yolox_tiny, yolox_nano, yolox_s, yolox_m or yolox_l.
./src/YOLOX-ROS/weights/tensorrt/convert.bash yolox_tiny
```

### Tensorflow Lite
```bash
cd ~/ros2_ws

# Download tflite Person Detection model: https://github.com/Kazuhito00/Person-Detection-using-RaspberryPi-CPU/
./src/YOLOX-ROS/weights/tflite/download_model.bash
```

### PINTO_model_zoo
- Support PINTO_model_zoo model
- Download model using the following script.
  - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/download_nano.sh
  - `curl -s https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/main/132_YOLOX/download_nano.sh | bash`

- ONNX model copy to weight dir
  - `cp resouces_new/saved_model_yolox_tiny_480x640/yolox_tiny_480x640.onnx ./src/YOLOX-ROS/weights/onnx/`

- Convert to TensorRT engine
  - `./src/YOLOX-ROS/weights/tensorrt/convert.bash yolox_tiny_480x640`

- tflite model copy to weight dir
  - `cp resouces_new/saved_model_yolox_tiny_480x640/model_float32.tflite ./src/YOLOX-ROS/weights/tflite/`


<br>

## Build

### OpenVINO

```bash
# build with openvino
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -DYOLOX_USE_OPENVINO=ON
```

### TensorRT

```bash
# build with tensorrt
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -DYOLOX_USE_TENSORRT=ON
```

### TFLite

**TFLite build**

https://www.tensorflow.org/lite/guide/build_cmake

Below is an example build script.
Please change `${WORKSPACE}` as appropriate for your envjazzyment.
```bash
export WORKSPACE=${HOME}/ws_tflite
mkdir -p ${WORKSPACE}
cd ${WORKSPACE}
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

```bash
colcon build --symlink-install \
   --cmake-args \
    -DYOLOX_USE_TFLITE=ON \
    -DTFLITE_LIB_PATH=${WORKSPACE}/tflite_build \
    -DTFLITE_INCLUDE_DIR=${WORKSPACE}/tensorflow_src/ \
    -DABSEIL_CPP_ICLUDE_DIR=${WORKSPACE}/tflite_build/abseil-cpp \
    -DFLATBUFFERS_INCLUDE_DIR=${WORKSPACE}/tflite_build/flatbuffers/include
```


### HailoRT

```bash
# build with hailort
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DYOLOX_USE_HAILORT=ON
```


<br>

## Run

### OpenVINO
```bash
ros2 launch yolox_ros_cpp yolox_openvino.launch.py

## run other model
# ros2 launch yolox_ros_cpp yolox_openvino.launch.py \
#     model_path:=install/yolox_ros_cpp/share/yolox_ros_cpp/weights/openvino/yolox_s.xml

## run PINTO_model_zoo model (version 0.1.0)
# ros2 launch yolox_ros_cpp yolox_openvino.launch.py \
#     model_path:=install/yolox_ros_cpp/share/yolox_ros_cpp/weights/onnx/yolox_tiny_480x640.onnx \
#     model_version:="0.1.0"
```

### TensorRT
```bash
ros2 launch yolox_ros_cpp yolox_tensorrt.launch.py

## run PINTO_model_zoo model (version 0.1.0)
# ros2 launch yolox_ros_cpp yolox_tensorrt.launch.py \
#     model_path:=install/yolox_ros_cpp/share/yolox_ros_cpp/weights/tensorrt/yolox_tiny_480x640.trt \
#     model_version:="0.1.0"

```

<!-- ### Jetson + TensorRT
Jetson docker container cannot display GUI.
If you want to show image with bounding box drawn, subscribe from host jetson or other PC.

```bash
# run yolox_tiny
ros2 launch yolox_ros_cpp yolox_tensorrt_jetson.launch.py
``` -->

### ONNXRuntime
```bash
# run yolox_tiny
ros2 launch yolox_ros_cpp yolox_onnxruntime.launch.py
```

### Tensorflow Lite
```bash
ros2 launch yolox_ros_cpp yolox_tflite.launch.py

# # run PINTO_model_zoo model (version 0.1.0)
# ros2 launch yolox_ros_cpp yolox_tflite.launch.py \
#     model_path:=install/yolox_ros_cpp/share/yolox_ros_cpp/weights/tflite/model_float32.tflite \
#     model_version:=0.1.0 \
#     num_classes:=80 \
#     is_nchw:=false
```

### HailoRT

```bash
ros2 launch yolox_ros_cpp yolox_hailort.launch.py
```

### Parameter

<details>
<summary>OpenVINO example</summary>

- `model_path`: ./src/YOLOX-ROS/weights/onnx/yolox_tiny.onnx
- `p6`: false
- `class_labels_path`: ""
  - if not set, use coco_names.
  - See [here](https://github.com/fateshelled/YOLOX-ROS/blob/dev_cpp/yolox_ros_cpp/yolox_ros_cpp/labels/coco_names.txt) for label format.
- `num_classes`: 80
- `model_version`: 0.1.1rc0
- `openvino_device`: AUTO
- `nms`: 0.45
- `imshow_isshow`: true
- `src_image_topic_name`: /image_raw
- `publish_image_topic_name`: /yolox/image_raw
- `publish_boundingbox_topic_name`: /yolox/bounding_boxes
- `use_bbox_ex_msgs`: false
- `publish_resized_image`: false

</details>


<details>
<summary>TensorRT example</summary>

- `model_path`: ../src/YOLOX-ROS/weights/tensorrt/yolox_tiny.trt
- `p6`: false
- `class_labels_path`: ""
- `num_classes`: 80
- `model_version`: 0.1.1rc0
- `tensorrt_device`: 0
- `conf`: 0.3
- `nms`: 0.45
- `imshow_isshow`: true
- `src_image_topic_name`: /image_raw
- `publish_image_topic_name`: /yolox/image_raw
- `publish_boundingbox_topic_name`: /yolox/bounding_boxes
- `use_bbox_ex_msgs`: false
- `publish_resized_image`: false

</details>

<details>
<summary>ONNXRuntime example</summary>


- `model_path`: ./src/YOLOX-ROS/weights/onnx/yolox_tiny.onnx
- `p6`: false
- `class_labels_path`: ""
- `num_classes`: 80
- `model_version`: 0.1.1rc0
- `onnxruntime_use_gpu`: true
- `onnxruntime_use_parallel`: false
- `onnxruntime_device_id`: 0
- `onnxruntime_inter_op_num_threads`: 1
  - if `onnxruntime_use_parallel` is true, the number of threads used to parallelize the execution of the graph (across nodes).
- `onnxruntime_intra_op_num_threads`: 1
  - the number of threads to use to run the model
- `conf`: 0.3
- `nms`: 0.45
- `imshow_isshow`: true
- `src_image_topic_name`: /image_raw
- `publish_image_topic_name`: /yolox/image_raw
- `publish_boundingbox_topic_name`: /yolox/bounding_boxes
- `use_bbox_ex_msgs`: false
- `publish_resized_image`: false

</details>

<details>
<summary>Tensorflow Lite example</summary>

- `model_path`: ./src/YOLOX-ROS/weights/tflite/model.tflite
- `p6`: false
- `is_nchw`: true
- `class_labels_path`: ""
- `num_classes`: 1
- `model_version`: 0.1.1rc0
- `tflite_num_threads`: 1
- `conf`: 0.3
- `nms`: 0.45
- `imshow_isshow`: true
- `src_image_topic_name`: /image_raw
- `publish_image_topic_name`: /yolox/image_raw
- `publish_boundingbox_topic_name`: /yolox/bounding_boxes
- `use_bbox_ex_msgs`: false
- `publish_resized_image`: false

</details>

## Reference
Reference from YOLOX demo code.
- https://github.com/Megvii-BaseDetection/YOLOX/blob/5183a6716404bae497deb142d2c340a45ffdb175/demo/OpenVINO/cpp/yolox_openvino.cpp
- https://github.com/Megvii-BaseDetection/YOLOX/tree/5183a6716404bae497deb142d2c340a45ffdb175/demo/TensorRT/cpp

HailoRT
- https://github.com/tier4/hailort-yoloXP
- https://github.com/hailo-ai/hailort