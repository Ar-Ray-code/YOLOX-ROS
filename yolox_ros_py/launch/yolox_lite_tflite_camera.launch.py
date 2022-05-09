import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from urllib.request import urlretrieve
import os
import time

def generate_launch_description():
    yolox_ros_share_dir = get_package_share_directory('yolox_ros_py')

    print("")
    print("-------------------------------------------------------")
    print("Warning : This model is based on YOLOX and is lightweight for RaspberryPi CPU operation. Detection other than human detection may not work correctly.")
    print("-------------------------------------------------------")
    print("")
    time.sleep(1)

    video_device = LaunchConfiguration('video_device', default='/dev/video0')
    video_device_arg = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/video0',
        description='Video device'
    )

    webcam = launch_ros.actions.Node(
        package="v4l2_camera", executable="v4l2_camera_node",
        parameters=[
            {"image_size": [640,480]},
            {"video_device": video_device},
        ],
    )

    yolox_tflite = launch_ros.actions.Node(
        package="yolox_ros_py", executable="yolox_tflite",output="screen",
        parameters=[
            {"model_path" : yolox_ros_share_dir+"/model.tflite"},
            {"conf" : 0.4},
            {"nms_th" : 0.5},
            {"input_shape/height" : 192},
            {"input_shape/width" : 192}
        ],
    )

    rqt_graph = launch_ros.actions.Node(
        package="rqt_graph", executable="rqt_graph",
    )

    return launch.LaunchDescription([
        video_device_arg,
        webcam,
        yolox_tflite,
        # rqt_graph
    ])