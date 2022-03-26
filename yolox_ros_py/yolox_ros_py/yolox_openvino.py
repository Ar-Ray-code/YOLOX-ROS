#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# ROS2 rclpy -- Ar-Ray-code 2021
import rclpy

import argparse
import logging as log

import cv2
import numpy as np

from openvino.inference_engine import IECore

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import multiclass_nms, demo_postprocess, vis

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

        WEIGHTS_PATH = '/home/ubuntu/ros2_ws/install/yolox_ros_py/share/yolox_ros_py/yolox_s.xml'

        self.declare_parameter('imshow_isshow',True)

        self.declare_parameter('model_path', WEIGHTS_PATH)
        self.declare_parameter('conf', 0.3)
        self.declare_parameter('device', "CPU")

        self.declare_parameter('image_size/width', 640)
        self.declare_parameter('image_size/height', 480)


        # =============================================================
        self.imshow_isshow = self.get_parameter('imshow_isshow').value

        self.model_path = self.get_parameter('model_path').value
        self.conf = self.get_parameter('conf').value
        self.device = self.get_parameter('device').value

        self.input_width = self.get_parameter('image_size/width').value
        self.input_height = self.get_parameter('image_size/height').value

        # ==============================================================

        print('Creating Inference Engine')
        ie = IECore()
        print(f'Reading the self.network: {self.model_path}')
        # (.xml and .bin files) or (.onnx file)

        self.net = ie.read_network(model=self.model_path)
        print('Configuring input and output blobs')
        # Get names of input and output blobs
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))

        # Set input and output precision manually
        self.net.input_info[self.input_blob].precision = 'FP32'
        self.net.outputs[self.out_blob].precision = 'FP16'

        print('Loading the model to the plugin')
        self.exec_net = ie.load_network(network=self.net, device_name=self.device)


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
            img_rgb = self.bridge.imgmsg_to_cv2(msg,"bgr8")

            origin_img = img_rgb
            _, _, h, w = self.net.input_info[self.input_blob].input_data.shape
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            image, ratio = preprocess(origin_img, (h, w))#, mean, std)

            res = self.exec_net.infer(inputs={self.input_blob: image})
            res = res[self.out_blob]

            # Predictions is result
            predictions = demo_postprocess(res, (h, w), p6=False)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4, None] * predictions[:, 5:]
            # print(scores)

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
            # boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

            # print(dets)
            if dets is not None:
                final_boxes = dets[:, :4]
                final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
                origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                                 conf=self.conf, class_names=COCO_CLASSES)
                
                # ==============================================================
            end_time = cv2.getTickCount()
            time_took = (end_time - start_time) / cv2.getTickFrequency()
                
            # rclpy log FPS
            self.get_logger().info(f'FPS: {1 / time_took}')
            
            try:
                bboxes = self.yolox2bboxes_msgs(dets[:, :4], final_scores, final_cls_inds, COCO_CLASSES, msg.header)
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
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(img_rgb,"bgr8"))

        except Exception as e:
            self.get_logger().info(f'Error: {e}')
            pass

    # origin_img is output

def ros_main(args = None) -> None:
    rclpy.init(args=args)

    yolox_ros_class = yolox_ros()
    rclpy.spin(yolox_ros_class)
    
    yolox_ros_class.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    ros_main()