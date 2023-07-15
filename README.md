# YOLOX-ROS

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) + ROS2 Humble demo

![yolox_s_result](https://github.com/Ar-Ray-code/RenderTexture2ROS2Image/blob/main/images_for_readme/unity-demo.gif?raw=true)

<div align="center">🔼 Unity + YOLOX-ROS Demo</div>

## Supported List


| Base            | ROS2 C++ |
| --------------- | -------- |
| TensorRT (CUDA) |  ✅       |
| OpenVINO        |  ✅       |
| ONNX Runtime    |  ✅       |
| TFLite          |  ✅       |


## Installation & Demo (C++)

Check [this URL](./yolox_ros_cpp/README.md).

<br>

## Topic

### Subscribe

- image_raw (`sensor_msgs/Image`)

### Publish

<!-- - yolox/image_raw : Resized image (`sensor_msgs/Image`) -->

- bounding_boxes: Output BoundingBoxes like darknet_ros_msgs (`bboxes_ex_msgs/BoundingBoxes`)

  ※ If you want to use `darknet_ros_msgs` , replace `bboxes_ex_msgs` with `darknet_ros_msgs`.

![yolox_topic](images_for_readme/yolox_topic.png)

<br>

## Parameters

- Check launch files.

<br>

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

<br>

## Contributors
- [Ar-Ray](https://github.com/Ar-Ray-code)
- [fateshelled](https://github.com/fateshelled)
- [Kazuhito00](https://github.com/Kazuhito00)
- [swiftfile](https://github.com/swiftfile)

<br>

## About writer

- Ar-Ray : Japanese student.
- Blog (Japanese) : https://ar-ray.hatenablog.com/
- Twitter : https://twitter.com/Ray255Ar
