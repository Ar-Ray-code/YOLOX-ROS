from setuptools import setup

import os
from glob import glob
from urllib.request import urlretrieve

package_name = 'yolox_ros_py'


YOLOX_S = 'yolox_s'
YOLOX_M = 'yolox_m'
YOLOX_L = 'yolox_l'
YOLOX_X = 'yolox_x'
YOLOX_DARKNET53 = 'yolox_darknet53'
YOLOX_NANO = 'yolox_nano'
YOLOX_TINY = 'yolox_tiny'

PTH = '.pth'
ONNX = '.onnx'

YOLOX_VERSION = '0.1.1rc0'

BASE_LINK = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/'+YOLOX_VERSION+'/'
# .pth
YOLOX_S_URL = BASE_LINK + YOLOX_S + PTH
YOLOX_M_URL = BASE_LINK + YOLOX_M + PTH
YOLOX_L_URL = BASE_LINK + YOLOX_L + PTH
YOLOX_X_URL = BASE_LINK + YOLOX_X + PTH
YOLOX_X_URL = BASE_LINK + YOLOX_DARKNET53 + PTH
YOLOX_NANO_URL = BASE_LINK + YOLOX_NANO + PTH
YOLOX_TINY_URL = BASE_LINK + YOLOX_TINY + PTH
# .onnx
YOLOX_S_ONNX_URL = BASE_LINK + YOLOX_S + ONNX
YOLOX_M_ONNX_URL = BASE_LINK + YOLOX_M + ONNX
YOLOX_L_ONNX_URL = BASE_LINK + YOLOX_L + ONNX
YOLOX_X_ONNX_URL = BASE_LINK + YOLOX_X + ONNX
YOLOX_X_ONNX_URL = BASE_LINK + YOLOX_DARKNET53 + ONNX
YOLOX_NANO_ONNX_URL = BASE_LINK + YOLOX_NANO + ONNX
YOLOX_TINY_ONNX_URL = BASE_LINK + YOLOX_TINY + ONNX

BASE_PATH = os.getcwd() + '/../weights/'
os.makedirs(BASE_PATH, exist_ok=True)
# .pth
YOLOX_S_PATH = BASE_PATH + YOLOX_S + PTH
YOLOX_M_PATH = BASE_PATH + YOLOX_M + PTH
YOLOX_L_PATH = BASE_PATH + YOLOX_L + PTH
YOLOX_X_PATH = BASE_PATH + YOLOX_X + PTH
YOLOX_DARKNET53_PATH = BASE_PATH + YOLOX_DARKNET53 + PTH
YOLOX_NANO_PATH = BASE_PATH + YOLOX_NANO + PTH
YOLOX_TINY_PATH = BASE_PATH + YOLOX_TINY + PTH
# .onnx
YOLOX_S_ONNX_PATH = BASE_PATH + YOLOX_S + ONNX
YOLOX_M_ONNX_PATH = BASE_PATH + YOLOX_M + ONNX
YOLOX_L_ONNX_PATH = BASE_PATH + YOLOX_L + ONNX
YOLOX_X_ONNX_PATH = BASE_PATH + YOLOX_X + ONNX
YOLOX_DARKNET53_ONNX_PATH = BASE_PATH + YOLOX_DARKNET53 + ONNX
YOLOX_NANO_ONNX_PATH = BASE_PATH + YOLOX_NANO + ONNX
YOLOX_TINY_ONNX_PATH = BASE_PATH + YOLOX_TINY + ONNX

# Download YOLOX-NANO Weights
if not os.path.exists(YOLOX_NANO_PATH):
    urlretrieve(YOLOX_NANO_URL, YOLOX_NANO_PATH)
# Download YOLOX-NANO ONNX
if not os.path.exists(YOLOX_NANO_ONNX_PATH):
    urlretrieve(YOLOX_NANO_ONNX_URL, YOLOX_NANO_ONNX_PATH)

# onnx
TFLITE_PATH = BASE_PATH + 'tflite/model.onnx'
if not os.path.exists(TFLITE_PATH):
    urlretrieve('https://github.com/Kazuhito00/Person-Detection-using-RaspberryPi-CPU/raw/main/model/model.onnx', TFLITE_PATH)

# tflite
TFLITE_PATH = BASE_PATH + 'tflite/model.tflite'
if not os.path.exists(TFLITE_PATH):
    urlretrieve('https://github.com/Kazuhito00/Person-Detection-using-RaspberryPi-CPU/raw/main/model/model.tflite', TFLITE_PATH)


setup(
    name=package_name,
    version='0.3.2',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('./launch/*.launch.py')),
        (os.path.join('share', package_name), glob('../weights/*.pth')),
        (os.path.join('share', package_name), glob('../weights/*.onnx')),
        (os.path.join('share', package_name), glob('../weights/*.tflite')),
        (os.path.join('share', package_name), glob('../weights/openvino/*')),
        (os.path.join('share', package_name), glob('../weights/onnx/*')),
        (os.path.join('share', package_name), glob('../weights/tflite/*')),
        (os.path.join('share', package_name), glob('../weights/tensorrt/*')),
        (os.path.join('share', package_name), glob('./exps/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Ar-Ray-code',
    author_email="ray255ar@gmail.com",
    maintainer='Ar-Ray-code',
    maintainer_email="ray255ar@gmail.com",
    description='YOLOX + ROS2 Foxy',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolox_ros = '+package_name+'.yolox_ros:ros_main',
            'yolox_openvino = '+package_name+'.yolox_openvino:ros_main',
            'yolox_onnx = '+package_name+'.yolox_onnx:ros_main',
            'yolox_tflite = '+package_name+'.yolox_tflite:ros_main',
        ],
    },
)

