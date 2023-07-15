# Jetson

## Environment

- Jetson AGX Orin
- Jetpack 5.1 (R35.1.0 @Ubuntu 20.04)
- Docker
- [TIER IV C1 Camera environment](https://github.com/tier4/tier4_automotive_hdr_camera) (When using C1 camera)

## Build & attach shell


```bash
cd ./YOLOX-ROS/yolox_ros_cpp/docker/jetson
bash setup.bash
```

## Build (in docker container)

```bash
source /dependencies/install/setup.bash
colcon build --cmake-args -DJETSON=ON
bash src/YOLOX-ROS/weights/onnx/download.bash yolox_tiny
bash src/YOLOX-ROS/weights/tensorrt/convert.bash yolox_tiny
```


## Run (in docker container)

```bash
source ./install/setup.bash
ros2 launch yolox_ros_cpp yolox_tensorrt.launch.py
```
