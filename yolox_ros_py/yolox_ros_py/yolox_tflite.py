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

class yolox_cpu(Node):
    def __init__(self):
        super().__init__('yolox_cpu')

        # パラメータ設定 ###################################################
        self.declare_parameter('model', './model/model.onnx')
        self.declare_parameter('score_th', 0.4)
        self.declare_parameter('nms_th', 0.5)
        self.declare_parameter('num_threads', None)
        self.declare_parameter('input_shape/height', 192)
        self.declare_parameter('input_shape/width', 192)

        self.declare_parameter('image_size/width', 640)
        self.declare_parameter('image_size/height', 480)

        self.declare_parameter('sensor_qos_mode', False)
        
        # パラメータ取得 ###################################################
        self.model_path = self.get_parameter('model').value
        self.score_th = self.get_parameter('score_th').value
        self.nms_th = self.get_parameter('nms_th').value
        self.num_threads = self.get_parameter('num_threads').value
        self.input_shape_h = self.get_parameter('input_shape/height').value
        self.input_shape_w = self.get_parameter('input_shape/width').value

        self.image_size_w = self.get_parameter('image_size/width').value
        self.image_size_h = self.get_parameter('image_size/height').value

        self.sensor_qos_mode = self.get_parameter('sensor_qos_mode').value

        self.input_shape = (self.input_shape_h, self.input_shape_w)

        self.bridge = CvBridge()

        self.yolox = Detector(
            model_path=self.model_path,
            input_shape=self.input_shape,
            score_th=self.score_th,
            nms_th=self.nms_th,
            providers=['CPUExecutionProvider'],
            num_threads=self.num_threads,
        )

        if (self.sensor_qos_mode):
            self.sub = self.create_subscription(Image,"image_raw",self.imageflow_callback, qos_profile_sensor_data)
        else:
            self.sub = self.create_subscription(Image,"image_raw",self.imageflow_callback, 10)

        self.pub_detection = self.create_publisher(
            BoundingBoxes,
            'detection',
            10
        )

    def imageflow_callback(self, msg):
        start = time.time()
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # resize
        image = cv2.resize(image, (self.image_size_w, self.image_size_h))
        bboxes, scores, class_ids = self.yolox.inference(image)
        elapsed_time = time.time() - start
        fps = 1 / elapsed_time

        self.image_h = image.shape[0]
        self.image_w = image.shape[1]

        # デバッグ描画
        debug_image = draw_debug(
            image,
            elapsed_time,
            self.score_th,
            bboxes,
            scores,
            class_ids,
        )

        # キー処理(ESC：終了) ##############################################
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            pass

        # 画面反映 #########################################################
        debug_image = cv2.resize(debug_image, (640, 480))
        cv2.imshow('debug_image', debug_image)

        # データ出力 #######################################################
        bboxes_msg = self.yolox2bboxes_msgs(bboxes, scores, class_ids, COCO_CLASSES, msg.header)

        self.pub_detection.publish(bboxes_msg)

        # print
        self.get_logger().info('fps: %.2f' % fps)

    def yolox2bboxes_msgs(self, bboxes, scores, cls, cls_names, img_header:Header):
        bboxes_msg = BoundingBoxes()
        bboxes_msg.header = img_header
        i = 0
        for bbox in bboxes:
            one_box = BoundingBox()
            # if < 0
            if bbox[0] < 0:
                bbox[0] = 0
            if bbox[1] < 0:
                bbox[1] = 0
            if bbox[2] < 0:
                bbox[2] = 0
            if bbox[3] < 0:
                bbox[3] = 0
            one_box.xmin = int(bbox[0])
            one_box.ymin = int(bbox[1])
            one_box.xmax = int(bbox[2])
            one_box.ymax = int(bbox[3])
            one_box.img_height = self.image_h
            one_box.img_width = self.image_w
            one_box.probability = float(scores[i])
            one_box.class_id = str(cls_names[int(cls[i])])
            bboxes_msg.bounding_boxes.append(one_box)
            i = i+1

        return bboxes_msg

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
