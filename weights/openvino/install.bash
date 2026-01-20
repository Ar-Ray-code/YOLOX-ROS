#!/bin/bash

# if $1 is empty
if [ -z "$1" ]; then
    echo "Usage: $0 <target-model>"
    echo "Target-Models :"
    echo "yolox_tiny, yolox_nano, yolox_s, yolox_m, yolox_l, yolox_darknet, yolox"
    exit 1
fi
MODEL=$1
MO=/opt/intel/openvino_2021/deployment_tools/model_optimizer/
ROS2_WS=~/ros2_ws

rm -rf ~/openvino_download
mkdir ~/openvino_download
cd ~/openvino_download
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/$MODEL.onnx
# if $1==yolox_tiny or $1==yolox_nano
cd $MO
if [ "$MODEL" == "yolox_tiny" ] || [ "$MODEL" == "yolox_nano" ]; then
    python3 mo.py --input_model ~/openvino_download/$MODEL.onnx --input_shape [1,3,416,416] --data_type FP16 --output_dir ~/openvino_download/
else
    python3 mo.py --input_model ~/openvino_download/$MODEL.onnx --input_shape [1,3,640,640] --data_type FP16 --output_dir ~/openvino_download/
fi
echo "=========================================="
echo "Model Optimizer finished"
echo "=========================================="

cp -r ~/openvino_download/* $ROS2_WS/src/YOLOX-ROS/weights/openvino/
cd $ROS2_WS
colcon build --symlink-install
