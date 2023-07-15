SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd ${SCRIPT_DIR}
mkdir ./tensorrt_dir
cp -r /usr/include/aarch64-linux-gnu/* ./tensorrt_dir/

docker build -t yolox_ros_jetson:latest -f ./dockerfile .
docker run -it --rm --network host --volume $HOME/ros2_ws:/root/ros2_ws --device /dev/video0:/dev/video0 --workdir /root/ros2_ws --runtime nvidia yolox_ros_jetson /bin/bash


# source /dependencies/install/setup.bash
# colcon build --cmake-args -DJETSON=ON
# bash src/YOLOX-ROS/weights/onnx/download.bash yolox_tiny
# bash src/YOLOX-ROS/weights/tensorrt/convert.bash yolox_tiny
