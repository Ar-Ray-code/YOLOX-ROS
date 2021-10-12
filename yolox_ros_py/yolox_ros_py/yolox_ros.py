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

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, setup_logger, vis

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from bboxes_ex_msgs.msg import BoundingBoxes
from bboxes_ex_msgs.msg import BoundingBox

# from darknet_ros_msgs.msg import BoundingBoxes
# from darknet_ros_msgs.msg import BoundingBox

class Predictor(object):
    def __init__(self, model, exp, cls_names=COCO_CLASSES, trt_file=None, decoder=None, device="cpu", fp16=False, legacy=False):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.threshold = exp.threshold
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.threshold, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

class yolox_ros(Node):
    def __init__(self) -> None:

        # ROS2 init
        super().__init__('yolox_ros')

        self.setting_yolox_exp()

        # if (self.imshow_isshow):
        #     cv2.namedWindow("YOLOX")
        
        self.bridge = CvBridge()
        
        self.pub = self.create_publisher(BoundingBoxes,"yolox/bounding_boxes", 10)
        self.pub_image = self.create_publisher(Image,"yolox/image_raw", 10)
        self.sub = self.create_subscription(Image,"image_raw",self.imageflow_callback, 10)

    def setting_yolox_exp(self) -> None:
        # set environment variables for distributed training
        
        # ==============================================================

        WEIGHTS_PATH = '../../weights/yolox_s.pth'

        self.declare_parameter('imshow_isshow',True)

        self.declare_parameter('yolo_type','yolox-s')
        self.declare_parameter('fuse',False)
        self.declare_parameter('trt', False)
        self.declare_parameter('fp16', False)
        self.declare_parameter('legacy', False)
        self.declare_parameter('device', "cpu")
        # self.declare_parameter('', 0)
        self.declare_parameter('ckpt', WEIGHTS_PATH)
        self.declare_parameter('conf', 0.3)

        # nmsthre -> threshold
        self.declare_parameter('threshold', 0.65)
        # --tsize -> resize
        self.declare_parameter('resize', 640)
        self.declare_parameter('image_size/width', 640)
        self.declare_parameter('image_size/height', 480)

        # =============================================================
        self.imshow_isshow = self.get_parameter('imshow_isshow').value

        yolo_type = self.get_parameter('yolo_type').value
        fuse = self.get_parameter('fuse').value
        trt = self.get_parameter('trt').value
        fp16 = self.get_parameter('fp16').value
        fp16 = self.get_parameter('fp16').value
        device = self.get_parameter('device').value
        #  = self.get_parameter('').value
        ckpt = self.get_parameter('ckpt').value
        conf = self.get_parameter('conf').value
        legacy = self.get_parameter('legacy').value
        threshold = self.get_parameter('threshold').value
        
        resize = self.get_parameter('resize').value
        self.input_width = self.get_parameter('image_size/width').value
        self.input_height = self.get_parameter('image_size/height').value

        # ==============================================================

        cudnn.benchmark = True

        exp = get_exp(None, yolo_type)

        BASE_PATH = os.getcwd()
        file_name = os.path.join(BASE_PATH, "YOLOX_PATH/")
        # os.makedirs(file_name, exist_ok=True)

        exp.test_conf = conf # test conf
        exp.threshold = threshold # nms threshold
        exp.test_size = (resize, resize) # Resize size

        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

        if device == "gpu":
            model.cuda()
            if fp16:
                model.half() 
        # torch.cuda.set_device()
        # model.cuda()
        model.eval()

        # about not trt
        if not trt:
            if ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        # about fuse
        if fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        # TensorRT
        if trt:
            assert not fuse, "TensorRT model is not support model fusing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None
        
        self.predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, device, fp16, legacy)

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
                    cv2.imshow("YOLOX",result_img_rgb)
                    cv2.waitKey(1)
                
            except:
                if (self.imshow_isshow):
                    cv2.imshow("YOLOX",img_rgb)
                    cv2.waitKey(1)
        except:
            pass

def ros_main(args = None) -> None:
    rclpy.init(args=args)

    yolox_ros_class = yolox_ros()
    rclpy.spin(yolox_ros_class)

    yolox_ros_class.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    ros_main()