# Copyright 2024 Ar-Ray-code
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'video_device',
            default_value='/dev/video0',
            description='input video source'
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value='./src/YOLOX-ROS/weights/hailort/yolox-s-opt-relu6-960x960-T4.hef',
            description='yolox model path.'
        ),
        DeclareLaunchArgument(
            'class_labels_path',
            default_value='',
            description='if use custom model, set class name labels. '
        ),
        DeclareLaunchArgument(
            'num_classes',
            default_value='8', # TODO: support 80 classes
            description='num classes.'
        ),
        DeclareLaunchArgument(
            'conf',
            default_value='0.30',
            description='yolox confidence threshold.'
        ),
        DeclareLaunchArgument(
            'nms',
            default_value='0.45',
            description='yolox nms threshold'
        ),
        DeclareLaunchArgument(
            'imshow_isshow',
            default_value='true',
            description=''
        ),
        DeclareLaunchArgument(
            'src_image_topic_name',
            default_value='/image_raw',
            description='topic name for source image'
        ),
        DeclareLaunchArgument(
            'publish_image_topic_name',
            default_value='/yolox/image_raw',
            description='topic name for publishing image with bounding box drawn'
        ),
        DeclareLaunchArgument(
            'publish_boundingbox_topic_name',
            default_value='/yolox/bounding_boxes',
            description='topic name for publishing bounding box message.'
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
                    'video_device': LaunchConfiguration('video_device'),
                    'image_size': [640, 480]
                }]),
            ComposableNode(
                package='yolox_ros_cpp',
                plugin='yolox_ros_cpp::YoloXNode',
                name='yolox_ros_cpp',
                parameters=[{
                    'model_path': LaunchConfiguration('model_path'),
                    'class_labels_path': LaunchConfiguration('class_labels_path'),
                    'num_classes': LaunchConfiguration('num_classes'),
                    'model_type': 'hailort',
                    'conf': LaunchConfiguration('conf'),
                    'nms': LaunchConfiguration('nms'),
                    'imshow_isshow': LaunchConfiguration('imshow_isshow'),
                    'src_image_topic_name': LaunchConfiguration('src_image_topic_name'),
                    'publish_image_topic_name': LaunchConfiguration('publish_image_topic_name'),
                    'publish_boundingbox_topic_name': LaunchConfiguration('publish_boundingbox_topic_name'),
                }],
            ),
        ],
        output='screen'
    )

    return launch.LaunchDescription(
        launch_args +
        [
            container
        ]
    )
