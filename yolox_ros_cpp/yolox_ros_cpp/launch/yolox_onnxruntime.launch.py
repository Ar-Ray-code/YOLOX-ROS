# Copyright 2023 Ar-Ray-code
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
            default_value='./src/YOLOX-ROS/weights/onnx/yolox_tiny.onnx',
            description='yolox model path.'
        ),
        DeclareLaunchArgument(
            'p6',
            default_value='false',
            description='with p6.'
        ),
        DeclareLaunchArgument(
            'class_labels_path',
            default_value='',
            description='if use custom model, set class name labels. '
        ),
        DeclareLaunchArgument(
            'num_classes',
            default_value='80',
            description='num classes.'
        ),
        DeclareLaunchArgument(
            'model_version',
            default_value='0.1.1rc0',
            description='yolox model version.'
        ),
        DeclareLaunchArgument(
            'onnxruntime/use_cuda',
            default_value='true',
            description='onnxruntime use cuda.'
        ),
        DeclareLaunchArgument(
            'onnxruntime/device_id',
            default_value='0',
            description='onnxruntime gpu device id.'
        ),
        DeclareLaunchArgument(
            'onnxruntime/use_parallel',
            default_value='false',
            description='if use_parallel is true, you can set inter_op_num_threads.'
        ),
        DeclareLaunchArgument(
            'onnxruntime/inter_op_num_threads',
            default_value='1'
        ),
        DeclareLaunchArgument(
            'onnxruntime/intra_op_num_threads',
            default_value='1',
            description='ontrols the number of threads to use to run the model.'
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
                    'p6': LaunchConfiguration('p6'),
                    'class_labels_path': LaunchConfiguration('class_labels_path'),
                    'num_classes': LaunchConfiguration('num_classes'),
                    'model_type': 'onnxruntime',
                    'model_version': LaunchConfiguration('model_version'),
                    'onnxruntime/use_cuda': LaunchConfiguration('onnxruntime/use_cuda'),
                    'onnxruntime/device_id': LaunchConfiguration('onnxruntime/device_id'),
                    'onnxruntime/use_parallel': LaunchConfiguration('onnxruntime/use_parallel'),
                    'onnxruntime/inter_op_num_threads': LaunchConfiguration('onnxruntime/inter_op_num_threads'),
                    'onnxruntime/intra_op_num_threads': LaunchConfiguration('onnxruntime/intra_op_num_threads'),
                    'conf': LaunchConfiguration('conf'),
                    'nms': LaunchConfiguration('nms'),
                    'imshow_isshow': LaunchConfiguration('imshow_isshow'),
                    'src_image_topic_name': LaunchConfiguration('src_image_topic_name'),
                    'publish_image_topic_name': LaunchConfiguration('publish_image_topic_name'),
                    'publish_boundingbox_topic_name': LaunchConfiguration('publish_boundingbox_topic_name'),
                }],
                ),
        ],
        output='screen',
    )

    return launch.LaunchDescription(
        launch_args +
        [
            container
        ]
    )
