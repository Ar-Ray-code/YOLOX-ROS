#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# ROS2 rclpy -- Ar-Ray-code 2021

import os
import time
from loguru import logger

import cv2
from numpy import empty

import torch
import torch.backends.cudnn as cudnn

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, setup_logger, vis

import rospy

from std_msgs.msg import Header
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
        return vis_res, bboxes, scores, cls, self.cls_names

class yolox_ros():
    def __init__(self) -> None:

        # ROS1 init

        self.setting_yolox_exp()

        self.bridge = CvBridge()
        
        self.pub = rospy.Publisher('yolox/bounding_boxes', BoundingBoxes)
        self.pub_image = rospy.Publisher("yolox/image_raw",Image)
        rospy.Subscriber("image_raw",Image,self.imageflow_callback)

        rospy.spin()

    def setting_yolox_exp(self) -> None:
        # set environment variables for distributed training
        
        # ==============================================================

        WEIGHTS_PATH = '../../weights/yolox_s.pth'

        # =============================================================
        self.imshow_isshow = rospy.get_param('imshow_isshow', True)

        yolo_type = rospy.get_param('~yolo_type', 'yolox-s')
        fuse = rospy.get_param('~fuse', False)
        trt = rospy.get_param('~trt', False)
        rank = rospy.get_param('~rank', 0)
        ckpt_file = rospy.get_param('~ckpt_file', WEIGHTS_PATH)
        conf = rospy.get_param('~conf', 0.3)
        nmsthre = rospy.get_param('~nmsthre', 0.65)
        img_size = rospy.get_param('~img_size', 640)
        self.input_width = rospy.get_param('~image_size/width', 360)
        self.input_height = rospy.get_param('~image_size/height', 240)

        # ==============================================================

        cudnn.benchmark = True

        exp = get_exp(None, yolo_type)

        BASE_PATH = os.getcwd()
        file_name = os.path.join(BASE_PATH, "YOLOX_PATH/")
        # os.makedirs(file_name, exist_ok=True)

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
            img_rgb = self.bridge.imgmsg_to_cv2(msg,"bgr8")
            img_rgb = cv2.resize(img_rgb,(self.input_width,self.input_height))

            outputs, img_info = self.predictor.inference(img_rgb)

            try:
                result_img_rgb, bboxes, scores, cls, cls_names = self.predictor.visual(outputs[0], img_info)
                bboxes = self.yolox2bboxes_msgs(bboxes, scores, cls, cls_names, msg.header)

                self.pub.publish(bboxes)
                self.pub_image.publish(self.bridge.cv2_to_imgmsg(img_rgb,"bgr8"))

                if (self.imshow_isshow):
                    cv2.imshow("YOLOX", result_img_rgb)
                    cv2.waitKey(10)
            except:
                if (self.imshow_isshow):
                    cv2.imshow("YOLOX",img_rgb)
                    cv2.waitKey(10)

        except:
            pass

def ros_main(args = None) -> None:
    rospy.init_node("yolox_ros",argv=args)
    yolox_ros()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ros_main()