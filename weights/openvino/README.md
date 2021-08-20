## YOLOX_Nano's weights file (OpenVINO)

- yolox_nano.bin
- yolox_nano.mapping
- yolox_nano.xml

## How to create and Install OpenVINO's weights

> Note : Setup Open VINO 2021 before run these commands.

```bash
export MODEL="yolox_tiny"
export MO=/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py

rm -rf ~/openvino_download
mkdir ~/openvino_download
cd ~/openvino_download
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/$MODEL.onnx
python3 $MO --input_model ./yolox_nano.onnx --input_shape [1,3,416,416] --data_type FP16 --output_dir ./
cp -r ./* ~/ros2_ws/src/YOLOX-ROS/weights/openvino/
cd ~/ros2_ws/
colcon build --symlink-install
```



