## YOLOX_Tiny's weights file (OpenVINO)

- yolox_tiny.bin ([Download bin](https://github.com/Ar-Ray-code/YOLOX-ROS/releases/download/v0.2.0/yolox_tiny.bin))
- yolox_tiny.xml ([Download xml](https://github.com/Ar-Ray-code/YOLOX-ROS/releases/download/v0.2.0/yolox_tiny.xml))

## How to create and Install OpenVINO's weights

> Note : Setup Open VINO 2021 before run these commands.

```bash
export MODEL="yolox_tiny"
export MO=/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py

rm -rf ~/openvino_download
mkdir ~/openvino_download
cd ~/openvino_download
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/$MODEL.onnx
python3 $MO --input_model ./$MODEL.onnx --input_shape [1,3,416,416] --data_type FP16 --output_dir ./
cp -r ./* ~/ros2_ws/src/YOLOX-ROS/weights/openvino/
cd ~/ros2_ws/
colcon build
```



