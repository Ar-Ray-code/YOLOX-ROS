import launch
import launch_ros.actions
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    yolox_ros_share_dir = get_package_share_directory('yolox_ros_py')

    yolox_ros = launch_ros.actions.Node(
        package="yolox_ros_py", executable="yolox_ros",
        parameters=[
            {"image_size/width": 640},
            {"image_size/height": 480},
            {"yolo_type" : 'yolox-s'},
            {"device" : 'gpu'},
            {"fp16" : True},
            {"fuse" : False},
            {"legacy" : False},
            {"trt" : False},
            {"ckpt" : yolox_ros_share_dir+"/yolox_s.pth"},
            {"conf" : 0.3},
            {"threshold" : 0.65},
            {"resize" : 640},
        ],
    )

    return launch.LaunchDescription([
        yolox_ros
    ])