# MIGraphX + ONNX Runtime (ROCm) build notes

This workspace uses ONNX Runtime built with the MIGraphX execution provider.
ROCm EP is not present in `/opt/onnxruntime-migraphx`, so MIGraphX is used as the GPU backend.

## Key findings

- `libyolox_cpp.so` links against `/opt/onnxruntime-migraphx/lib/libonnxruntime.so.1`.
- That onnxruntime build exposes `MIGraphXExecutionProvider` (not `ROCMExecutionProvider`).
- MIGraphX tries to save compiled models; if a cache path is not set, it can fail with:
  - `migraphx_save: ... Failure opening file: ""/....mxr`
- To avoid this, ensure `ORT_MIGRAPHX_MODEL_CACHE_PATH` (and/or `ORT_MIGRAPHX_CACHE_PATH`) is set.

## Prerequisites (Ubuntu 24.04 + ROCm)

### Install Radeon driver + ROCm

If you already installed `amdgpu-install_7.1.x`, install the ROCm usecase:

```bash
sudo amdgpu-install --list-usecase
sudo amdgpu-install -y --usecase=graphics,rocm
sudo reboot
```

After reboot, allow GPU compute access and reboot again:

```bash
sudo usermod -a -G render,video $USER
sudo reboot
```

Verify ROCm is visible:

```bash
/opt/rocm/bin/rocminfo | less
hipcc --version
```

### Install MIGraphX packages

```bash
sudo apt update
sudo apt install -y migraphx migraphx-dev half
dpkg -l | egrep 'migraphx|half'
```

Check the MIGraphX install path (used as `--migraphx_home` below):

```bash
dpkg -L migraphx-dev | head -n 50
```

## Build and install ONNX Runtime (MIGraphX EP)

### Build

ROCm EP was removed in onnxruntime 1.23+, so use MIGraphX EP.

```bash
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --build_shared_lib --parallel \
  --use_migraphx --migraphx_home /opt/rocm \
  --update --build --skip_tests
```

### Install

```bash
cd build/Linux/Release
sudo cmake --install . --prefix /opt/onnxruntime-migraphx
echo /opt/onnxruntime-migraphx/lib | sudo tee /etc/ld.so.conf.d/onnxruntime.conf
sudo ldconfig
ldconfig -p | grep onnxruntime
```

## Build (colcon)

Example build command (matches local setup with `/opt/onnxruntime-migraphx`):

```
colcon build --symlink-install \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DYOLOX_USE_ONNXRUNTIME=ON \
    -DONNXRUNTIME_MIGRAPHX=ON \
    -DCMAKE_CXX_FLAGS="-I/opt/onnxruntime-migraphx/include" \
    -DCMAKE_EXE_LINKER_FLAGS="-L/opt/onnxruntime-migraphx/lib -lonnxruntime" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/onnxruntime-migraphx/lib -lonnxruntime"
```

## Runtime cache path

Set a writable cache directory before running:

```
export ORT_MIGRAPHX_MODEL_CACHE_PATH=/tmp/onnxruntime-migraphx-cache
export ORT_MIGRAPHX_CACHE_PATH=/tmp/onnxruntime-migraphx-cache
```

## Run example

```
source install/setup.bash
ros2 run yolox_ros_cpp yolox_ros_cpp_node --ros-args \
  -p model_path:=./src/YOLOX-ROS/weights/onnx/yolox_tiny.onnx \
  -p model_type:=onnxruntime \
  -p use_gpu:=true
```
