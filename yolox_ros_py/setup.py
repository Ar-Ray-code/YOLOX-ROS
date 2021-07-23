import os
from glob import glob
from setuptools import setup
from urllib.request import urlretrieve

package_name = 'yolox_ros_py'

YOLOX_S_WEIGHTS = 'yolox_s.pth'
YOLOX_M_WEIGHTS = 'yolox_m.pth'
YOLOX_L_WEIGHTS = 'yolox_l.pth'
YOLOX_X_WEIGHTS = 'yolox_x.pth'
YOLOX_DARKNET53_WEIGHTS = 'yolox_darknet53.pth'
YOLOX_NANO_WEIGHTS = 'yolox_nano.pth'
YOLOX_TINY_WEIGHTS = 'yolox_tiny.pth'

BASE_LINK = 'https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/'
YOLOX_S_WEIGHTS_URL = BASE_LINK + YOLOX_S_WEIGHTS
YOLOX_M_WEIGHTS_URL = BASE_LINK + YOLOX_M_WEIGHTS
YOLOX_L_WEIGHTS_URL = BASE_LINK + YOLOX_L_WEIGHTS
YOLOX_X_WEIGHTS_URL = BASE_LINK + YOLOX_X_WEIGHTS
YOLOX_DARKNET53_WEIGHTS_URL = BASE_LINK + YOLOX_DARKNET53_WEIGHTS
YOLOX_NANO_WEIGHTS_URL = BASE_LINK + YOLOX_NANO_WEIGHTS
YOLOX_TINY_WEIGHTS_URL = BASE_LINK + YOLOX_TINY_WEIGHTS

BASE_PATH = os.getcwd() + '/../weights/'
os.makedirs(BASE_PATH, exist_ok=True)
YOLOX_S_WEIGHTS_PATH = BASE_PATH + YOLOX_S_WEIGHTS
YOLOX_M_WEIGHTS_PATH = BASE_PATH + YOLOX_M_WEIGHTS
YOLOX_L_WEIGHTS_PATH = BASE_PATH + YOLOX_L_WEIGHTS
YOLOX_X_WEIGHTS_PATH = BASE_PATH + YOLOX_X_WEIGHTS
YOLOX_DARKNET53_WEIGHTS_PATH = BASE_PATH + YOLOX_DARKNET53_WEIGHTS
YOLOX_NANO_WEIGHTS_PATH = BASE_PATH + YOLOX_NANO_WEIGHTS
YOLOX_TINY_WEIGHTS_PATH = BASE_PATH + YOLOX_TINY_WEIGHTS

if not os.path.exists(YOLOX_S_WEIGHTS_PATH):
    urlretrieve(YOLOX_S_WEIGHTS_URL, YOLOX_S_WEIGHTS_PATH)

# if not os.path.exists(YOLOX_M_WEIGHTS_PATH):
#     urlretrieve(YOLOX_M_WEIGHTS_URL, YOLOX_M_WEIGHTS_PATH)
    
# if not os.path.exists(YOLOX_L_WEIGHTS_PATH):
#     urlretrieve(YOLOX_L_WEIGHTS_URL, YOLOX_L_WEIGHTS_PATH)

# if not os.path.exists(YOLOX_X_WEIGHTS_PATH):
#     urlretrieve(YOLOX_X_WEIGHTS_URL, YOLOX_X_WEIGHTS_PATH)

# if not os.path.exists(YOLOX_DARKNET53_WEIGHTS_PATH):
#     urlretrieve(YOLOX_DARKNET53_WEIGHTS_URL, YOLOX_DARKNET53_WEIGHTS_PATH)

# if not os.path.exists(YOLOX_NANO_WEIGHTS_PATH):
#     urlretrieve(YOLOX_NANO_WEIGHTS_URL, YOLOX_NANO_WEIGHTS_PATH)

# if not os.path.exists(YOLOX_TINY_WEIGHTS_PATH):
#     urlretrieve(YOLOX_TINY_WEIGHTS_URL, YOLOX_TINY_WEIGHTS_PATH)

setup(
    name=package_name,
    version='1.0.0',
    packages=[],
    py_modules= [
        'scripts.yolox_ros',
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Ar-Ray-code',
    author_email="ray255ar@gmail.com",
    maintainer='user',
    maintainer_email="ray255ar@gmail.com",
    keywords=['ROS', 'ROS2'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='YOLOX + ROS2 Foxy',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolox_ros = scripts.yolox_ros:ros_main',
        ],
    },
    data_files=[
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob('../weights/*')),
    ],
)