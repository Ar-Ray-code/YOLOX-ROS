import launch
import launch_ros.actions
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

from urllib.request import urlretrieve
import os

def generate_launch_description():
    yolox_ros_share_dir = get_package_share_directory('yolox_ros_py')

    webcam = launch_ros.actions.Node(
        package="v4l2_camera", executable="v4l2_camera_node",
        parameters=[
            {"image_size": [640,480]},
        ],
    )
    yolox_openvino = launch_ros.actions.Node(
        package="yolox_ros_py", executable="yolox_openvino",output="screen",
        parameters=[
            {"image_size/width": 640},
            {"image_size/height": 480},
            {"device" : 'CPU'},
            {"model_path" : yolox_ros_share_dir+"/yolox_nano.onnx"},
            {"conf" : 0.3},
        ],
    )

    rqt_graph = launch_ros.actions.Node(
        package="rqt_graph", executable="rqt_graph",
    )

    return launch.LaunchDescription([
        webcam,
        yolox_openvino,
        # rqt_graph
    ])