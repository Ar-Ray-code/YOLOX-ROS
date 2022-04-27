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

from detector import Detector
from demo import draw_debug

# ROS2 =====================================
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D

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
        
        # パラメータ取得 ###################################################
        self.model_path = self.get_parameter('model').value
        self.score_th = self.get_parameter('score_th').value
        self.nms_th = self.get_parameter('nms_th').value
        self.num_threads = self.get_parameter('num_threads').value
        self.input_shape_h = self.get_parameter('input_shape/height').value
        self.input_shape_w = self.get_parameter('input_shape/width').value

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

        self.sub_image = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10
        )

        self.pub_detection = self.create_publisher(
            Detection2DArray,
            'detection',
            10
        )

    def image_callback(self, msg):
        start = time.time()
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # resize
        # image = cv2.resize(image, (self.width, self.height))
        bboxes, scores, class_ids = self.yolox.inference(image)
        elapsed_time = time.time() - start
        fps = 1 / elapsed_time

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
        msg = Detection2DArray()
        msg.header = Header()
        msg.header.stamp = msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'detection'
        msg.detections = []
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            detection = Detection2D()

            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            size_w = bbox[2] - bbox[0]
            size_h = bbox[3] - bbox[1]

            msg.detections.append(detection)

            detection.bbox.center.x = float(center_x)
            detection.bbox.center.y = float(center_y)
            detection.bbox.size_x = float(size_w)
            detection.bbox.size_y = float(size_h)
            # if person -> add
            if class_id == 0:
                msg.detections.append(detection)
        self.pub_detection.publish(msg)

        # print
        print('elapsed_time: {:.3f}[ms], fps: {:.1f}'.format(elapsed_time * 1000, fps))
        for detection in msg.detections:
            print('detection:', detection.bbox.center.x, detection.bbox.center.y, detection.bbox.size_x, detection.bbox.size_y)

    def __del__(self):
        cv2.destroyAllWindows()
        self.sub_image.destroy()
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
