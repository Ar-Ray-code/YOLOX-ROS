#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# ROS2 rclpy -- Ar-Ray-code 2021

import argparse
import os
import time
from loguru import logger

import cv2

import torch
import torch.backends.cudnn as cudnn

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, setup_logger, vis

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from bboxes_ex_msgs.msg import BoundingBoxes
from bboxes_ex_msgs.msg import BoundingBox

# from darknet_ros_msgs.msg import BoundingBoxes
# from darknet_ros_msgs.msg import BoundingBox

class Predictor(object):
    def __init__(self, model, exp, cls_names=COCO_CLASSES, trt_file=None, decoder=None):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {'id': 0}
        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None

        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        img_info['raw_img'] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info['ratio'] = ratio
        img = torch.from_numpy(img).unsqueeze(0).cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                        outputs, self.num_classes, self.confthre, self.nmsthre
                    )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info['ratio']
        img = img_info['raw_img']
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res ,bboxes

class yolox_ros(Node):

    def __init__(self) -> None:

        # ROS2 init
        super().__init__('yolox_ros')

        cv2.namedWindow("window")

        self.setting_yolox_exp()
        self.bridge = CvBridge()
        
        self.pub = self.create_publisher(BoundingBoxes,"yololx/bounding_boxes", 10)
        self.sub = self.create_subscription(Image,"image_raw",self.imageflow_callback, 10)

    def setting_yolox_exp(self):
        # set environment variables for distributed training
        yolo_type = 'yolox-s'
        cudnn.benchmark = True
        fuse = False
        trt=False
        rank = 0
        ckpt_file = '/home/ubuntu/ros2_ws/src/yolox_ros/weights/yolox_s.pth.tar'
        conf = 0.3
        nmsthre = 0.65
        img_size = 640

        exp = get_exp(None, yolo_type)

        file_name = os.path.join(exp.output_dir, "./")
        os.makedirs(file_name, exist_ok=True)

        exp.test_conf = conf # test conf
        exp.nmsthre = nmsthre # nms threshold
        exp.test_size = (img_size, img_size) # Resize size

        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

        torch.cuda.set_device(rank)
        model.cuda(rank)
        model.eval()

        if not trt:
            logger.info("loading checkpoint")
            loc = "cuda:{}".format(rank)
            ckpt = torch.load(ckpt_file, map_location=loc)
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        if fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        # TensorRT
        if trt:
            assert (not fuse),\
                "TensorRT model is not support model fusing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(trt_file), (
                "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            )
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None

        self.predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder)
        

    def imageflow_callback(self,msg:Image):
        try:
            img_rgb = self.bridge.imgmsg_to_cv2(msg,"bgr8")

            outputs, img_info = self.predictor.inference(img_rgb)
            # print(outputs)
            result_img_rgb, bboxes = self.predictor.visual(outputs[0], img_info)
            print(bboxes)
            cv2.imshow("window",result_img_rgb)
            cv2.waitKey(1)
            # output_img = self.bridge
        except Exception as e:
            print(e)


def ros_main(args = None):
    rclpy.init(args=args)

    yolox_ros_class = yolox_ros()
    rclpy.spin(yolox_ros_class)

    yolox_ros_class.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    ros_main()