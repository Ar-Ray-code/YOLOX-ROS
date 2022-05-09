#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# ROS2 rclpy -- Ar-Ray-code 2021
import rclpy

import cv2
import numpy as np
import copy

from openvino.inference_engine import IECore

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import multiclass_nms, demo_postprocess, vis

from .yolox_ros_py_utils.utils import yolox_py

# ROS2 =====================================
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from rclpy.qos import qos_profile_sensor_data

from bboxes_ex_msgs.msg import BoundingBoxes
from bboxes_ex_msgs.msg import BoundingBox

# from darkself.net_ros_msgs.msg import BoundingBoxes
# from darkself.net_ros_msgs.msg import BoundingBox

class yolox_ros(yolox_py):
    def __init__(self) -> None:

        # ROS2 init
        super().__init__('yolox_ros', load_params=True)

        self.setting_yolox_exp()

        if (self.imshow_isshow):
            cv2.namedWindow("YOLOX")
        
        self.bridge = CvBridge()
        
        self.pub = self.create_publisher(BoundingBoxes,"bounding_boxes", 10)
        self.sub = self.create_subscription(Image,"image_raw",self.imageflow_callback, self.qos_image_sub)

    def setting_yolox_exp(self) -> None:
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

    def imageflow_callback(self,msg:Image) -> None:
        try:
            # fps start
            start_time = cv2.getTickCount()
            bboxes = BoundingBoxes()
            origin_img = self.bridge.imgmsg_to_cv2(msg,"bgr8")
            # deep copy
            nodetect_image = copy.deepcopy(origin_img)

            # origin_img = img_rgb
            _, _, h, w = self.net.input_info[self.input_blob].input_data.shape
            image, ratio = preprocess(origin_img, (h, w))

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
            boxes_xyxy /= ratio
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
                bboxes = self.yolox2bboxes_msgs(dets[:, :4], final_scores, final_cls_inds, COCO_CLASSES, msg.header,  origin_img)
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