# YOLOX-ROS

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) + ROS2 Foxy (cuda 10.2)

__NVIDIA Graphics is required ❗❗❗__


![yolox_s_result](images_for_readme/yolox_s_result.png)

Japanese Reference (Plan to post)：[Qiita](https://qiita.com/Ar-Ray)

## Supported List

| Base        | ROS1 C++ | ROS1 Python | ROS2 C++ | ROS2 Python |
| ----------- | -------- | ----------- | -------- | ---------- |
| MegEngine   |          |             |          |            |
| ONNXRuntime |          |             |          |            |
| TensorRT    |          |             |          |            |
| CUDA        |          | ✅           |          | ✅          |
| ncnn        |          |             |          |            |
| OpenVINO    |          |             |          |            |
| Others(?)   |          |             |          |            |

## Requirements (Python)

- ROS2 Foxy
- CUDA 10.2
- OpenCV 4.5.1
- Python 3.8 (Ubuntu 20.04 Default)
- Torch '1.9.0+cu102 (Install with pytorch)
- cuDNN 7.6.5 (Install with pytorch)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- TensorRT : is not supported
- WebCamera : v4l2_camera

## Requirements (C++)

- C++ is not supported

## Installation

Install the dependent packages based on all tutorials.

### STEP 1 : CUDA Installation 

- [CUDA-10.2-toolkit](https://developer.nvidia.com/cuda-10.2-download-archive)

### STEP 2 : YOLOX Quick-start

[YOLOX Quick-start (Python)](https://github.com/Megvii-BaseDetection/YOLOX#quick-start)

```
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

### STEP 3 : Install YOLOX-ROS

```bash
source /opt/ros/foxy/setup.bash
sudo apt install ros-foxy-v4l2-camera
git clone --recursive https://github.com/Ar-Ray-code/yolox_ros.git ~/ros2_ws/src/yolox_ros/
cd ~/ros2_ws
colcon build --symlink-install # weights files will be installed automatically.
```

## Demo

Connect your web camera.

```bash
source ~/ros2_ws/install/setup.bash
# Example 1 : YOLOX-s demo
ros2 launch yolox_ros_py demo_yolox_s.launch.py
# Example 2 : YOLOX-l demo
ros2 launch yolox_ros_py demo_yolox_l.launch.py
```



---

## Topic
### Subscribe

- image_raw (`sensor_msgs/Image`)

### Publish

- yolox/image_raw : Resized image (`sensor_msgs/Image`)

- yololx/bounding_boxes : Output BoundingBoxes like darknet_ros_msgs (`bboxes_ex_msgs/BoundingBoxes`)

  ※ If you want to use `darknet_ros_msgs` , replace `bboxes_ex_msgs` with `darknet_ros_msgs`.

![yolox_topic](images_for_readme/yolox_topic.png)

## Parameters : default

- image_size/width: 640
- image_size/height: 480
- yolo_type : 'yolox-s'
- fuse : False
- trt : False
- rank : 0
- ckpt_file : `/home/ubuntu/ros2_ws/src/yolox_ros/weights/yolox_s.pth.tar`
- conf : 0.3
- nmsthre : 0.65
- img_size : 640

---

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

## About writer

- Ar-Ray : Japanese student.
- Blog (Japanese) : https://ar-ray.hatenablog.com/
- Twitter : https://twitter.com/Ray255Ar
