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
            default_value="./install/yolox_ros_cpp/share/yolox_ros_cpp/weights/tensorrt/yolox_nano.trt",
            description="yolox model path."
        ),
        DeclareLaunchArgument(
            "class_labels_path",
            default_value="''",
            description="if use custom model, set class name labels. "
        ),
        DeclareLaunchArgument(
            "model_version",
            default_value="0.1.1rc0",
            description="yolox model version."
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
        DeclareLaunchArgument(
            "imshow_isshow",
            default_value="false",
            description=""
        ),
        DeclareLaunchArgument(
            "src_image_topic_name",
            default_value="/image_raw",
            description="topic name for source image"
        ),
        DeclareLaunchArgument(
            "publish_image_topic_name",
            default_value="/yolox/image_raw",
            description="topic name for publishing image with bounding box drawn"
        ),
        DeclareLaunchArgument(
            "publish_boundingbox_topic_name",
            default_value="/yolox/bounding_boxes",
            description="topic name for publishing bounding box message."
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
                            "class_labels_path": LaunchConfiguration("class_labels_path"),
                            "model_type": "tensorrt",
                            "model_version": LaunchConfiguration("model_version"),
                            "tensorrt/device": 0,
                            "conf": LaunchConfiguration("conf"),
                            "nms": LaunchConfiguration("nms"),
                            "imshow_isshow": LaunchConfiguration("imshow_isshow"),
                            "src_image_topic_name": LaunchConfiguration("src_image_topic_name"),
                            "publish_image_topic_name": LaunchConfiguration("publish_image_topic_name"),
                            "publish_boundingbox_topic_name": LaunchConfiguration("publish_boundingbox_topic_name"),
                        }],
                    ),
                ],
                output='screen',
        )

    return launch.LaunchDescription(
        launch_args + [container]
    )
