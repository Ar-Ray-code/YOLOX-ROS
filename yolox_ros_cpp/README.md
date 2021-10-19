# YOLOX-ROS-CPP
## Usage
### Requirements
- OpenVINO 2021
- ROS2 Foxy

* YOLOX is not require.

### Install YOLOX-CPP
```bash
source /opt/intel/openvino_2021/bin/setupvars.sh
cd ~/ros2_ws/src
git clone https://github.com/fateshelled/YOLOX-ROS -b dev_cpp
# Download onnx file and Convert to IR format.
./YOLOX-ROS/weights/openvino/install.bash yolox_nano
```

### DEMO
```bash

```

### Parameter: default
- image_size/width: 416
- image_size/height: 416
- model_path: /home/ubuntu/ros2_ws/src/YOLOX-ROS/weights/openvino/yolox_nano.xml
- device: CPU
- conf: 0.3
- nms: 0.45
- imshow_isshow: true


### Reference
Reference from YOLOX OpenVINO demo code.
- https://github.com/Megvii-BaseDetection/YOLOX/blob/5183a6716404bae497deb142d2c340a45ffdb175/demo/OpenVINO/cpp/yolox_openvino.cpp
