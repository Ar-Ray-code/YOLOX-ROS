#!/bin/bash

# if $1 is empty
if [ -z "$1" ]; then
    echo "Usage: $0 <target-model>"
    echo "Target-Models :"
    echo "yolox_tiny, yolox_nano, yolox_s, yolox_m, yolox_l, yolox_darknet, yolox"
    exit 1
fi
MODEL=$1
SCRIPT_DIR=$(cd $(dirname $0); pwd)

ONNX_MODEL_PATH=$SCRIPT_DIR/../../onnx/$MODEL.onnx
if [ ! -e $ONNX_MODEL_PATH ]; then
    $SCRIPT_DIR/../../onnx/download.bash $MODEL
fi

trtexec --onnx=$SCRIPT_DIR/../../onnx/$MODEL.onnx \
        --saveEngine=$SCRIPT_DIR/$MODEL.trt --fp16 --verbose --workspace=100000

echo "=========================================="
echo "Finished model convert"
echo "=========================================="

