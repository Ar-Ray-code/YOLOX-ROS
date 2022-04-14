#!/bin/bash

# if $1 is empty
if [ -z "$1" ]; then
    echo "Usage: $0 <target-model> <workspace>"
    echo "Target-Models : yolox_tiny, yolox_nano, yolox_s, yolox_m, yolox_l"
    echo "WORKSPACE : GPU memory workspace. Default 16."
    exit 1
fi

MODEL=$1
TRT_WORKSPACE=$2
if [ -z "$2" ]; then
    TRT_WORKSPACE=16
fi

SCRIPT_DIR=$(cd $(dirname $0); pwd)

echo "Model Name: ${MODEL}"
echo "Workspace size: ${TRT_WORKSPACE}"
echo ""

ONNX_MODEL_PATH=$SCRIPT_DIR/../onnx/$MODEL.onnx
if [ ! -e $ONNX_MODEL_PATH ]; then
    $SCRIPT_DIR/../onnx/download.bash $MODEL
fi

if [ ! -e $ONNX_MODEL_PATH ]; then
    echo "[ERROR] Not Found ${ONNX_MODEL_PATH}"
    echo "[ERROR] Please check target model name."
    exit 1
fi

/usr/src/tensorrt/bin/trtexec \
    --onnx=$SCRIPT_DIR/../onnx/$MODEL.onnx \
    --saveEngine=$SCRIPT_DIR/$MODEL.trt \
    --fp16 --verbose --workspace=$((1<<$TRT_WORKSPACE))
