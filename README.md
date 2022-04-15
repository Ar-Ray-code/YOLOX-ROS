# YOLOX-ROS

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)+ROS2 Foxy

![yolox_s_result](https://github.com/Ar-Ray-code/RenderTexture2ROS2Image/blob/main/images_for_readme/unity-demo.gif?raw=true)

<div align="center">🔼 Unity + YOLOX-ROS Demo</div>

## Supported List

| Base            | ROS2 C++ | ROS2 Python |
| --------------- | -------- | ----------- |
| CPU             |           | ✅           |
| CUDA            |           | ✅           |
| CUDA (FP16)     |           | ✅           |
| TensorRT (CUDA) |  ✅       |              |
| OpenVINO        |  ✅       | ✅           |

### Raspberry Pi4 🍓 + NCS2 + YOLOX-ROS

Good news for studets❗🍓

Check [GitHub Wiki](https://github.com/Ar-Ray-code/YOLOX-ROS/wiki/YOLOX-ROS---Raspbian-(NCS2)) to try YOLOX-ROS.

## Installation & Demo
<details>
<summary>Python (PyTorch)</summary>

## Requirements

- ROS2 Foxy
- OpenCV 4
- Python 3.8 (Ubuntu 20.04 Default)
- PyTorch >= v1.7
- [YOLOX v0.2.0](https://github.com/Megvii-BaseDetection/YOLOX)
- [bbox_ex_msgs](https://github.com/Ar-Ray-code/bbox_ex_msgs)

## Installation

Install the dependent packages based on all tutorials.

### STEP 1 : Download from GitHub

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Ar-Ray-code/yolox_ros.git --recursive
```

### STEP 2 : YOLOX Installation (yolox_rps_py)

```bash
bash YOLOX-ROS/yolox_ros_py/install_yolox_py.bash
```

### STEP 3 : Install YOLOX-ROS

```bash
source /opt/ros/foxy/setup.bash
sudo apt install ros-foxy-v4l2-camera
colcon build --symlink-install # weights files will be installed automatically.
```

### (Step 3) Using CUDA

If you have NVIDIA Graphics, you can run YOLOX-ROS on GPU.

**Additional installing lists**

- NVIDIA Graphics Driver
- CUDA toolkit (11.0)
- torch+cuda

### Step : Demo

Connect your web camera.

```bash
source /opt/ros/foxy/setup.bash
source ~/ros2_ws/install/local_setup.bash
ros2 launch yolox_ros_py yolox_s_cpu.launch.py
# ros2 launch yolox_ros_py yolox_s.launch.py # <- GPU
```

</details>

## C++

Check [this URL](https://github.com/Ar-Ray-code/YOLOX-ROS/tree/main/yolox_ros_cpp).


## Topic
### Subscribe

- image_raw (`sensor_msgs/Image`)

### Publish

- yolox/image_raw : Resized image (`sensor_msgs/Image`)

- yololx/bounding_boxes : Output BoundingBoxes like darknet_ros_msgs (`bboxes_ex_msgs/BoundingBoxes`)

  ※ If you want to use `darknet_ros_msgs` , replace `bboxes_ex_msgs` with `darknet_ros_msgs`.

![yolox_topic](images_for_readme/yolox_topic.png)

## Parameters 

- Check launch files.

## Composition

- Supports C++ only.

## YOLOX-ROS + Unity

Check this repository ([RenderTexture2ROS2Image](https://github.com/Ar-Ray-code/RenderTexture2ROS2Image)).

## Reference

![](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets/logo.png)

- [YOLOX (GitHub)](https://github.com/Megvii-BaseDetection/YOLOX)

```
@article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

## Contributors
- [Ar-Ray](https://github.com/Ar-Ray-code)
- [fateshelled](https://github.com/fateshelled)

## About writer

- Ar-Ray : Japanese student.
- Blog (Japanese) : https://ar-ray.hatenablog.com/
- Twitter : https://twitter.com/Ray255Ar
