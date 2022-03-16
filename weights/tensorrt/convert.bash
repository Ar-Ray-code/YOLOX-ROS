#!/bin/bash

CURDIR=`pwd`
# if $1 is empty
if [ -z "$1" ]; then
    echo "Usage: $0 <target-model> <workspace> <use-trtexec>"
    echo "Target-Models :"
    echo "yolox_tiny, yolox_nano, yolox_s, yolox_m, yolox_l"
    echo "WORKSAPCE : GPU memory workspace."
    echo "Use-trtexex : If not use trtexec, set 0. "
    exit 1
fi

TRT_WORKSPACE=$2
if [ -z "$2" ]; then
    TRT_WORKSPACE=32
fi

# use trtexec
USE_TRTEXEC=$3
if [ -z "$3" ]; then
    USE_TRTEXEC=1
fi

MODEL=$1
SCRIPT_DIR=$(cd $(dirname $0); pwd)
YOLOX_DIR=/workspace/YOLOX

echo $MODEL

if [ $USE_TRTEXEC = 1 ]; then
    ONNX_MODEL_PATH=$SCRIPT_DIR/../onnx/$MODEL.onnx
    if [ ! -e $ONNX_MODEL_PATH ]; then
        $SCRIPT_DIR/../onnx/download.bash $MODEL
    fi

    trtexec \
        --onnx=$SCRIPT_DIR/../onnx/$MODEL.onnx \
        --saveEngine=$SCRIPT_DIR/$MODEL.trt \
        --fp16 --verbose --workspace=$((1<<$TRT_WORKSPACE))
else
    EXPS="$MODEL"
    if [ "$MODEL" = "yolox_nano" ]; then
        if [ "$YOLOX_VERSION" = "0.1.0" -o "$YOLOX_VERSION" = "0.1.1rc0"]; then
            EXPS="nano"
        fi
    fi

    PYTORCH_MODEL_PATH=$SCRIPT_DIR/../pytorch/$MODEL.pth
    if [ ! -e $PYTORCH_MODEL_PATH ]; then
        $SCRIPT_DIR/../pytorch/download.bash $MODEL
    fi

    cd $YOLOX_DIR
    if [ "$YOLOX_VERSION" = "0.2.0" ]; then
        python3 tools/trt.py -f exps/default/$EXPS.py \
                            -c $PYTORCH_MODEL_PATH \
                            -w $TRT_WORKSPACE
    else
        python3 tools/trt.py -f exps/default/$EXPS.py \
                            -c $PYTORCH_MODEL_PATH
    fi
    mv YOLOX_outputs/$MODEL/model_trt.engine $SCRIPT_DIR/$MODEL.engine
    mv YOLOX_outputs/$MODEL/model_trt.pth $SCRIPT_DIR/$MODEL.pth
fi
cd $CURDIR