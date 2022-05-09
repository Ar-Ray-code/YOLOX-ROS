#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# ROS2 rclpy -- Ar-Ray-code 2022
import argparse
import os

import cv2
import numpy as np

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis

from .yolox_ros_py_utils.utils import yolox_py

# ROS2 =====================================
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from bboxes_ex_msgs.msg import BoundingBoxes
from bboxes_ex_msgs.msg import BoundingBox

from rclpy.qos import qos_profile_sensor_data

# from darkself.net_ros_msgs.msg import BoundingBoxes
# from darkself.net_ros_msgs.msg import BoundingBox

class yolox_ros(yolox_py):
    def __init__(self) -> None:

        # ROS2 init
        super().__init__('yolox_ros', load_params=True)

        if (self.imshow_isshow):
            cv2.namedWindow("YOLOX")
        
        self.bridge = CvBridge()
        
        self.pub = self.create_publisher(BoundingBoxes,"bounding_boxes", 10)
        self.sub = self.create_subscription(Image,"image_raw",self.imageflow_callback, self.qos_image_sub)

    def imageflow_callback(self,msg:Image) -> None:
        try:
            # fps start
            start_time = cv2.getTickCount()
            bboxes = BoundingBoxes()
            origin_img = self.bridge.imgmsg_to_cv2(msg,"bgr8")

            # preprocess
            img, self.ratio = preprocess(origin_img, self.input_shape)

            session = onnxruntime.InferenceSession(self.model_path)

            ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
            output = session.run(None, ort_inputs)
            
            predictions = demo_postprocess(output[0], self.input_shape, p6=self.with_p6)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
            boxes_xyxy /= self.ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.nms_th, score_thr=self.conf)
            if dets is not None:
                self.final_boxes, self.final_scores, self.final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                origin_img = vis(origin_img, self.final_boxes, self.final_scores, self.final_cls_inds,
                         conf=self.conf, class_names=COCO_CLASSES)

            end_time = cv2.getTickCount()
            time_took = (end_time - start_time) / cv2.getTickFrequency()

            # rclpy log FPS
            self.get_logger().info(f'FPS: {1 / time_took}')
            
            try:
                bboxes = self.yolox2bboxes_msgs(dets[:, :4], self.final_scores, self.final_cls_inds, COCO_CLASSES, msg.header, origin_img)
                if (self.imshow_isshow):
                    cv2.imshow("YOLOX",origin_img)
                    cv2.waitKey(1)

            except:
                if (self.imshow_isshow):
                    cv2.imshow("YOLOX",origin_img)
                    cv2.waitKey(1)

            self.pub.publish(bboxes)

        except Exception as e:
            self.get_logger().info(f'Error: {e}')
            pass

def ros_main(args = None):
    rclpy.init(args=args)
    ros_class = yolox_ros()

    try:
        rclpy.spin(ros_class)
    except KeyboardInterrupt:
        pass
    finally:
        ros_class.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    ros_main()