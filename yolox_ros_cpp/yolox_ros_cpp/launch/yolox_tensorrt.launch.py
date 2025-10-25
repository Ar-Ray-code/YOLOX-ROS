import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch.actions import SetEnvironmentVariable
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
            default_value='/home/triton/Documents/cv-dev/TR-Autonomy-2024-2025/src/TR-YOLOX-ROS/weights/tensorrt/modelInt8_clocks.engine',
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
            'tensorrt_device',
            default_value='0',
            description='GPU index. Set in string type. ex 0'
        ),
        DeclareLaunchArgument(
            'model_version',
            default_value='0.1.1rc0',
            description='yolox model version.'
        ),
        DeclareLaunchArgument(
            'conf',
            default_value='0.7',
            description='yolox confidence threshold.'
        ),
        DeclareLaunchArgument(
            'nms',
            default_value='0.45',
            description='yolox nms threshold'
        ),
        DeclareLaunchArgument(
            'imshow_isshow',
            default_value='false',
            description=''
        ),
        DeclareLaunchArgument(
            'src_image_topic_name',
            default_value='/camera/image',
            description='topic name for source image'
        ),
        DeclareLaunchArgument(
            'publish_image_topic_name',
            default_value='/yolox/image_raw',
            description='topic name for publishing image with bounding box drawn'
        ),
        DeclareLaunchArgument(
            'publish_boundingbox_topic_name',
            default_value='/detections',
            description='topic name for publishing bounding box message.'
        ),
        DeclareLaunchArgument(
            'use_bbox_ex_msgs',
            default_value='false',
            description='use BoundingBoxArray message type.'
        ),
        DeclareLaunchArgument(
            'publish_resized_image',
            default_value='false',
            description='use BoundingBoxArray message type.'
        ),
    ]
    SetEnvironmentVariable(
            name='RCLCPP_EXECUTOR_THREAD_COUNT',
            value='2'
        ),
    container = ComposableNodeContainer(
        name='yolox_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='yolox_ros_cpp',
                plugin='yolox_ros_cpp::YoloXNode',
                name='yolox_ros_cpp',
                parameters=[{
                    'model_path': LaunchConfiguration('model_path'),
                    'p6': LaunchConfiguration('p6'),
                    'class_labels_path': LaunchConfiguration('class_labels_path'),
                    'num_classes': LaunchConfiguration('num_classes'),
                    'model_type': 'tensorrt',
                    'model_version': LaunchConfiguration('model_version'),
                    'tensorrt_device': LaunchConfiguration('tensorrt_device'),
                    'conf': LaunchConfiguration('conf'),
                    'nms': LaunchConfiguration('nms'),
                    'imshow_isshow': LaunchConfiguration('imshow_isshow'),
                    'src_image_topic_name': LaunchConfiguration('src_image_topic_name'),
                    'publish_image_topic_name': LaunchConfiguration('publish_image_topic_name'),
                    'publish_boundingbox_topic_name': LaunchConfiguration('publish_boundingbox_topic_name'),
                    'publish_resized_image': LaunchConfiguration('publish_resized_image'),
                    'use_bbox_ex_msgs': LaunchConfiguration('use_bbox_ex_msgs'),
                }],
            ),
        ],
        output='screen',
    )

    return launch.LaunchDescription(
        launch_args + [container]
    )
