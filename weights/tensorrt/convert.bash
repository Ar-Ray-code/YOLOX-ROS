#!/bin/bash

CURDIR=`pwd`
# if $1 is empty
if [ -z "$1" ]; then
    echo "Usage: $0 <target-model> <workspace>"
    echo "Target-Models :"
    echo "yolox_tiny, yolox_nano, yolox_s, yolox_m, yolox_l"
    echo "WORKSAPCE : GPU memory workspace."
    exit 1
fi

TRT_WORKSPACE=$2
if [ -z "$2" ]; then
    TRT_WORKSPACE=32
fi

MODEL=$1
SCRIPT_DIR=$(cd $(dirname $0); pwd)

echo $MODEL

ONNX_MODEL_PATH=$SCRIPT_DIR/../onnx/$MODEL.onnx
if [ ! -e $ONNX_MODEL_PATH ]; then
    $SCRIPT_DIR/../onnx/download.bash $MODEL
fi

trtexec \
    --onnx=$SCRIPT_DIR/../onnx/$MODEL.onnx \
    --saveEngine=$SCRIPT_DIR/$MODEL.trt \
    --fp16 --verbose --workspace=$((1<<$TRT_WORKSPACE))

cd $CURDIR