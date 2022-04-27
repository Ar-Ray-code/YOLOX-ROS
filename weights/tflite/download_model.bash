SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wget https://github.com/Kazuhito00/Person-Detection-using-RaspberryPi-CPU/raw/main/model/model.onnx -O ${SCRIPT_DIR}/model.onnx
wget https://github.com/Kazuhito00/Person-Detection-using-RaspberryPi-CPU/raw/main/model/model.tflite -O ${SCRIPT_DIR}/model.tflite