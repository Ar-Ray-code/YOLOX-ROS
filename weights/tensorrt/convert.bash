#!/bin/bash

CURDIR=`pwd`
# if $1 is empty
if [ -z "$1" ]; then
    echo "Usage: $0 <target-model>"
    echo "Target-Models :"
    echo "yolox_tiny, yolox_nano, yolox_s, yolox_m, yolox_l"
    exit 1
fi

# trt.py in YOLOX main branch has new vairable.
TRT_WORKSPACE=32
if [ -z "$2" ]; then
    TRT_WORKSPACE=$2
fi

MODEL=$1
SCRIPT_DIR=$(cd $(dirname $0); pwd)

echo $MODEL
if [ "$MODEL" = "yolox_nano" ]; then
    EXPS="nano"
else
    EXPS="$MODEL"
fi

PYTORCH_MODEL_PATH=$SCRIPT_DIR/../pytorch/$MODEL.pth
if [ ! -e $PYTORCH_MODEL_PATH ]; then
    $SCRIPT_DIR/../pytorch/download.bash $MODEL
fi

cd /workspace/YOLOX
python3 tools/trt.py -f exps/default/$EXPS.py \
                     -c $PYTORCH_MODEL_PATH
# python3 tools/trt.py -f exps/default/$EXPS.py \
#                      -c $PYTORCH_MODEL_PATH \
#                      -w $TRT_WORKSPACE
cp -r YOLOX_outputs $SCRIPT_DIR
cd $CURDIR