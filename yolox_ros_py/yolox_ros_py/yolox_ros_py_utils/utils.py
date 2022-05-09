import numpy as np

from bboxes_ex_msgs.msg import BoundingBoxes
from bboxes_ex_msgs.msg import BoundingBox

# from darknet_ros_msgs.msg import BoundingBoxes
# from darknet_ros_msgs.msg import BoundingBox

from rclpy.node import Node
from std_msgs.msg import Header
from rclpy.qos import qos_profile_sensor_data

from loguru import logger
import sys

class yolox_py(Node):
    def __init__(self, name, load_params=True):
        super().__init__(name)
        if load_params:
            self.parameter_ros2()

    def yolox2bboxes_msgs(self, bboxes, scores, cls, cls_names, img_header: Header, image: np.ndarray) -> BoundingBoxes:
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

            if "bboxes_ex_msgs" in sys.modules:
                one_box.img_height = image.shape[0]
                one_box.img_width = image.shape[1]
            else:
                pass
            
            one_box.probability = float(scores[i])
            one_box.class_id = str(cls_names[int(cls[i])])
            bboxes_msg.bounding_boxes.append(one_box)
            i = i+1

        return bboxes_msg

    def parameter_ros2(self):
        # パラメータ設定 ###################################################
        self.declare_parameter('model_path', './model/model.onnx')
        # self.declare_parameter('score_th', 0.4)
        self.declare_parameter('nms_th', 0.5)
        self.declare_parameter('conf', 0.3)
        self.declare_parameter('device', "CPU")

        self.declare_parameter('num_threads', None)
        self.declare_parameter('input_shape/height', 416)
        self.declare_parameter('input_shape/width', 416)
        self.declare_parameter('imshow_isshow',True)
        self.declare_parameter('with_p6', False)

        self.declare_parameter('sensor_qos_mode', False)
        
        # パラメータ取得 ###################################################
        self.model_path = self.get_parameter('model_path').value
        # self.score_th = self.get_parameter('score_th').value
        self.nms_th = self.get_parameter('nms_th').value
        self.conf = self.get_parameter('conf').value
        self.device = self.get_parameter('device').value
        
        self.num_threads = self.get_parameter('num_threads').value
        self.input_shape_h = self.get_parameter('input_shape/height').value
        self.input_shape_w = self.get_parameter('input_shape/width').value
        self.imshow_isshow = self.get_parameter('imshow_isshow').value
        self.with_p6 = self.get_parameter('with_p6').value

        self.sensor_qos_mode = self.get_parameter('sensor_qos_mode').value

        if self.sensor_qos_mode:
            self.qos_image_sub = qos_profile_sensor_data
        else:
            self.qos_image_sub = 10

        self.input_shape = (self.input_shape_h, self.input_shape_w)


        # ==============================================================
        logger.info("parameters -------------------------------------------------")
        logger.info("model_path: {}".format(self.model_path))
        logger.info("nms_th (ONNX): {}".format(self.nms_th))
        logger.info("conf: {}".format(self.conf))
        logger.info("device: {}".format(self.device))
        logger.info("num_threads: {}".format(self.num_threads))
        logger.info("input_shape: {}".format(self.input_shape))
        logger.info("imshow_isshow: {}".format(self.imshow_isshow))
        logger.info("with_p6 (ONNX): {}".format(self.with_p6))
        logger.info("sensor_qos_mode: {}".format(self.sensor_qos_mode))
        logger.info("-------------------------------------------------------------")
        # ==============================================================
