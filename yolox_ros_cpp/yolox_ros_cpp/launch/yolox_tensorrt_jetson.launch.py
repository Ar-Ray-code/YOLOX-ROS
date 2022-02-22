import os
import sys
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():

    launch_args = [
        DeclareLaunchArgument(
            "video_device",
            default_value="/dev/video0",
            description="input video source"
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value="./install/yolox_ros_cpp/share/yolox_ros_cpp/weights/tensorrt/YOLOX_outputs/yolox_nano/model_trt.engine",
            description="yolox model path."
        ),
        DeclareLaunchArgument(
            "image_size/height",
            default_value="416",
            description="model input image height."
        ),
        DeclareLaunchArgument(
            "image_size/width",
            default_value="416",
            description="model input image width."
        ),
        DeclareLaunchArgument(
            "conf",
            default_value="0.30",
            description="yolox confidence threshold."
        ),
        DeclareLaunchArgument(
            "nms",
            default_value="0.45",
            description="yolox nms threshold"
        ),
    ]
    container = ComposableNodeContainer(
                name='yolox_container',
                namespace='',
                package='rclcpp_components',
                executable='component_container',
                composable_node_descriptions=[
                    ComposableNode(
                        package='v4l2_camera',
                        plugin='v4l2_camera::V4L2Camera',
                        name='v4l2_camera',
                        parameters=[{
                            "video_device": LaunchConfiguration("video_device"), 
                            "image_size": [640,480]
                        }]
                    ),
                    ComposableNode(
                        package='yolox_ros_cpp',
                        plugin='yolox_ros_cpp::YoloXNode',
                        name='yolox_ros_cpp',
                        parameters=[{
                            "model_path": LaunchConfiguration("model_path"),
                            "model_type": "tensorrt",
                            "device": "'0'",
                            "image_size/height": LaunchConfiguration("image_size/height"),
                            "image_size/width": LaunchConfiguration("image_size/width"),
                            "conf": LaunchConfiguration("conf"),
                            "nms": LaunchConfiguration("nms"),
                            "imshow_isshow": False,
                            "src_image_topic_name": "/image_raw",
                            "publish_image_topic_name": "/yolox/image_raw",
                            "publish_boundingbox_topic_name": "/yolox/bounding_boxes",
                        }],
                    ),
                ],
                output='screen',
        )

    return launch.LaunchDescription(
        launch_args + [container]
    )