import os
import sys
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    yolox_ros_share_dir = get_package_share_directory('yolox_ros_cpp')
    yolox_param_yaml = os.path.join(yolox_ros_share_dir, "param", "tiny_trtexec.yaml")

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
                            "image_size": [640,480]
                        }]),
                    ComposableNode(
                        package='yolox_ros_cpp',
                        plugin='yolox_ros_cpp::YoloXNode',
                        name='yolox_ros_cpp',
                        parameters=[yolox_param_yaml],
                        )
                ],
                output='screen',
        )

    rqt_graph = launch_ros.actions.Node(
        package="rqt_graph", executable="rqt_graph",
    )

    return launch.LaunchDescription([
        container,
        rqt_graph,
    ])