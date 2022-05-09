import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_ros


def generate_launch_description():
    gazebo_ros_share_dir = get_package_share_directory('gazebo_ros')
    yolox_ros_share_dir = get_package_share_directory('yolox_ros_py')
    gazebo_plugins_share_dir = get_package_share_directory('gazebo_plugins')

    world = os.path.join(
        gazebo_plugins_share_dir,
        'worlds',
        'gazebo_ros_camera_demo.world'
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share_dir, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share_dir, 'launch', 'gzclient.launch.py')
        )
    )
    
    yolox_onnx = launch_ros.actions.Node(
        package="yolox_ros_py", executable="yolox_onnx",output="screen",
        parameters=[
            {"input_shape/width": 416},
            {"input_shape/height": 416},

            {"with_p6" : False},
            {"model_path" : yolox_ros_share_dir+"/yolox_nano.onnx"},
            {"conf" : 0.3},
            {"sensor_qos_mode" : True},
        ],
        remappings=[
            ("/image_raw", "/demo_cam/camera1/image_raw"),
        ],
    )

    return LaunchDescription([
        gzserver_cmd,
        gzclient_cmd,
        yolox_onnx,
    ])
