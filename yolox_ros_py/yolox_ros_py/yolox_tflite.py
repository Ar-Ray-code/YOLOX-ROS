#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------
# Ar-Ray-code 2022
# -------------------

# Env(CPU): Raspberry Pi Bullseye, Ubuntu 20
# Env(ROS2): ROS2-Foxy, Galactic

# input /image_raw(Sensor_msgs/Image)
# output /detection(Vision_msgs/Detection2DArray)

# run --------------------------------------------------
# terminal1: ros2 run v4l2_camera v4l2_camera_node
# terminal2: python3 ./demo_ros2.py
# ------------------------------------------------------

import time
import cv2

from .module_tflite.detector import Detector
from .module_tflite.demo import draw_debug
from .yolox_ros_py_utils.utils import yolox_py

# ROS2 =====================================
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from yolox.data.datasets import COCO_CLASSES

from rclpy.qos import qos_profile_sensor_data

from bboxes_ex_msgs.msg import BoundingBoxes
from bboxes_ex_msgs.msg import BoundingBox

class yolox_cpu(yolox_py):
    def __init__(self):
        super().__init__('yolox_cpu', load_params=True)

        self.bridge = CvBridge()

        self.yolox = Detector(
            model_path=self.model_path,
            input_shape=self.input_shape,
            score_th=self.conf,
            nms_th=self.nms_th,
            providers=['CPUExecutionProvider'],
            num_threads=self.num_threads,
        )

        self.sub = self.create_subscription(Image,"image_raw",self.imageflow_callback, self.qos_image_sub)
        self.pub_detection = self.create_publisher(BoundingBoxes, 'bounding_boxes', 10)

    def imageflow_callback(self, msg):
        start = time.time()
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        bboxes, scores, class_ids = self.yolox.inference(image)
        elapsed_time = time.time() - start
        fps = 1 / elapsed_time

        self.image_h = image.shape[0]
        self.image_w = image.shape[1]

        # デバッグ描画
        debug_image = draw_debug(
            image,
            elapsed_time,
            self.conf,
            bboxes,
            scores,
            class_ids,
        )

        cv2.imshow('debug_image', debug_image)
        cv2.waitKey(1)

        bboxes_msg = self.yolox2bboxes_msgs(bboxes, scores, class_ids, COCO_CLASSES, msg.header, image)

        self.pub_detection.publish(bboxes_msg)
        self.get_logger().info('fps: %.2f' % fps)

    def __del__(self):
        cv2.destroyAllWindows()
        self.pub_detection.destroy()
        super().destroy_node()

def ros_main(args = None):
    rclpy.init(args=args)
    ros_class = yolox_cpu()

    try:
        rclpy.spin(ros_class)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
    
if __name__ == "__main__":
    ros_main()
