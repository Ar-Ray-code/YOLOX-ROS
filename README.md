# YOLOX-ROS

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) + ROS2 Foxy (cuda 10.2)

![yolox_s_result](images_for_readme/yolox_s_result.png)

Japanese Reference (Plan to post)：[Qiita](https://qiita.com/Ar-Ray)

## Requirements (Python)

- ROS2 Foxy
- CUDA 10.2
- OpenCV 4.5.1
- Python 3.8 (Ubuntu 20.04 Default)
- Torch '1.9.0+cu102 (Install with pytorch)
- cuDNN 7.6.5 (Install with pytorch)
- TensorRT : is not supported
- WebCamera : v4l2_camera

## Requirements (C++)

- C++ is not supported

## Installation

Install the dependent packages based on all tutorials.

- [CUDA-10.2-toolkit](https://developer.nvidia.com/cuda-10.2-download-archive)
- [YOLOX Quick-start (Python)](https://github.com/Megvii-BaseDetection/YOLOX#quick-start)
- Download weights file from [URL (yolox_s)](https://megvii-my.sharepoint.cn/personal/gezheng_megvii_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fgezheng%5Fmegvii%5Fcom%2FDocuments%2Fmodel%5Fcheckpoints%2FYOLOX%2Fyolox%5Fs%2Fyolox%5Fs%2Epth%2Etar&parent=%2Fpersonal%2Fgezheng%5Fmegvii%5Fcom%2FDocuments%2Fmodel%5Fcheckpoints%2FYOLOX%2Fyolox%5Fs&originalPath=aHR0cHM6Ly9tZWd2aWktbXkuc2hhcmVwb2ludC5jbi86dTovZy9wZXJzb25hbC9nZXpoZW5nX21lZ3ZpaV9jb20vRVc2MmdtTzJ2bk5OczVucHhqenVuVndCOXAzMDdxcXlnYUNrWGRUTzg4QkxVZz9ydGltZT1tb0N0T3VOTTJVZw) and place it in this file.

after, run these commands

```bash
source /opt/ros/foxy/setup.bash
sudo apt install ros-foxy-v4l2-camera
git clone https://github.com/Ar-Ray-code/yolox_ros.git ~/ros2_ws/src/yolox_ros/
cd ~/ros2_ws
colcon build --symlink-install
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