#!/bin/bash

function convert {
    # if $1 is empty
    if [ -z "$1" ]; then
        echo "Usage: $0 <target-model>"
        echo "Target-Models : yolox_tiny, yolox_nano, yolox_s, yolox_m, yolox_l"
        return
    fi

    MODEL=$1
    SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

    echo "Model Name: ${MODEL}"
    echo ""

    ONNX_MODEL_PATH=$SCRIPT_DIR/../onnx/$MODEL.onnx
    if [ ! -e $ONNX_MODEL_PATH ]; then
        $SCRIPT_DIR/../onnx/download.bash $MODEL
    fi

    if [ ! -e $ONNX_MODEL_PATH ]; then
        echo "[ERROR] Not Found ${ONNX_MODEL_PATH}"
        echo "[ERROR] Please check target model name."
        return
    fi

    /usr/src/tensorrt/bin/trtexec \
        --onnx=$SCRIPT_DIR/../onnx/$MODEL.onnx \
        --saveEngine=$SCRIPT_DIR/$MODEL.trt \
        --fp16 --verbose
}

convert $1
