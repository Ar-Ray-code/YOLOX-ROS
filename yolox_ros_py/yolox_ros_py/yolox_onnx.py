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

# ROS2 =====================================
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from bboxes_ex_msgs.msg import BoundingBoxes
from bboxes_ex_msgs.msg import BoundingBox

# from darkself.net_ros_msgs.msg import BoundingBoxes
# from darkself.net_ros_msgs.msg import BoundingBox

class yolox_ros(Node):
    def __init__(self) -> None:

        # ROS2 init
        super().__init__('yolox_ros')

        self.setting_yolox_exp()

        if (self.imshow_isshow):
            cv2.namedWindow("YOLOX")
        
        self.bridge = CvBridge()
        
        self.pub = self.create_publisher(BoundingBoxes,"yolox/bounding_boxes", 10)
        self.pub_image = self.create_publisher(Image,"yolox/image_raw", 10)
        self.sub = self.create_subscription(Image,"image_raw",self.imageflow_callback, 10)

    def setting_yolox_exp(self) -> None:
        # set environment variables for distributed training
        
        # ==============================================================

        ONNX_PATH = './install/yolox_ros_py/share/yolox_ros_py/yolox_nano.onnx'

        self.declare_parameter('imshow_isshow',True)

        self.declare_parameter('model_path', ONNX_PATH)
        self.declare_parameter('conf', 0.3)
        self.declare_parameter('with_p6', False)
        self.declare_parameter('input_shape/width', 416)
        self.declare_parameter('input_shape/height', 416)

        self.declare_parameter('image_size/width', 640)
        self.declare_parameter('image_size/height', 480)

        # =============================================================
        self.imshow_isshow = self.get_parameter('imshow_isshow').value

        self.model_path = self.get_parameter('model_path').value
        self.conf = self.get_parameter('conf').value

        self.input_width = self.get_parameter('image_size/width').value
        self.input_height = self.get_parameter('image_size/height').value
        self.input_shape_w = self.get_parameter('input_shape/width').value
        self.input_shape_h = self.get_parameter('input_shape/height').value

        # ==============================================================
        self.with_p6 = self.get_parameter('with_p6').value

        self.get_logger().info('model_path: {}'.format(self.model_path))
        self.get_logger().info('conf: {}'.format(self.conf))
        self.get_logger().info('input_shape: {}'.format((self.input_shape_w, self.input_shape_h)))
        self.get_logger().info('image_size: {}'.format((self.input_width, self.input_height)))


        self.input_shape = (self.input_shape_h, self.input_shape_w)


    def yolox2bboxes_msgs(self, bboxes, scores, cls, cls_names, img_header:Header):
        bboxes_msg = BoundingBoxes()
        bboxes_msg.header = img_header
        i = 0
        for bbox in bboxes:
            one_box = BoundingBox()
            one_box.xmin = int(bbox[0])
            one_box.ymin = int(bbox[1])
            one_box.xmax = int(bbox[2])
            one_box.ymax = int(bbox[3])
            one_box.probability = float(scores[i])
            one_box.class_id = str(cls_names[int(cls[i])])
            bboxes_msg.bounding_boxes.append(one_box)
            i = i+1

        return bboxes_msg

    def imageflow_callback(self,msg:Image) -> None:
        try:
            # fps start
            start_time = cv2.getTickCount()
            bboxes = BoundingBoxes()
            origin_img = self.bridge.imgmsg_to_cv2(msg,"bgr8")
            # resize
            img = cv2.resize(origin_img, (self.input_width, self.input_height))

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
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=self.conf)
            if dets is not None:
                self.final_boxes, self.final_scores, self.final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                origin_img = vis(origin_img, self.final_boxes, self.final_scores, self.final_cls_inds,
                         conf=self.conf, class_names=COCO_CLASSES)
            
            end_time = cv2.getTickCount()
            time_took = (end_time - start_time) / cv2.getTickFrequency()
                
            # rclpy log FPS
            self.get_logger().info(f'FPS: {1 / time_took}')
            
            try:
                bboxes = self.yolox2bboxes_msgs(dets[:, :4], self.final_scores, self.final_cls_inds, COCO_CLASSES, msg.header)
                # self.get_logger().info(f'bboxes: {bboxes}')
                if (self.imshow_isshow):
                    cv2.imshow("YOLOX",origin_img)
                    cv2.waitKey(1)
                
            except:
                # self.get_logger().info('No object detected')
                if (self.imshow_isshow):
                    cv2.imshow("YOLOX",origin_img)
                    cv2.waitKey(1)

            self.pub.publish(bboxes)
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(origin_img,"bgr8"))

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