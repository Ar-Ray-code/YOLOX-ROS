SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR
YOLOX_S_OPENVINO_TAR="yolox_s_openvino.tar.gz"
# $SCRIPT_DIR/../weights/openvino/yolox_s.xml is exists
if [ -f "$SCRIPT_DIR/yolox_s.xml" ]; then
    echo "yolox_s.xml is exists"
    exit 0
else
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/$YOLOX_S_OPENVINO_TAR -O $SCRIPT_DIR/$YOLOX_S_OPENVINO_TAR
    tar -xvf $SCRIPT_DIR/$YOLOX_S_OPENVINO_TAR -C $SCRIPT_DIR/
    rm $YOLOX_S_OPENVINO_TAR
fi
echo "Done. Please run `colcon build` to install."