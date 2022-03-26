SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR
if [ -e "$SCRIPT_DIR/YOLOX/" ]; then
    echo "YOLOX is exists"
    exit 0
fi
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -v -e .
mv setup.py _setup.py
mv setup.cfg _setup.cfg