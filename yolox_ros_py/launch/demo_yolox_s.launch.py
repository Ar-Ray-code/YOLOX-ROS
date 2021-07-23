import launch
import launch_ros.actions
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    yolox_ros_share_dir = get_package_share_directory('yolox_ros_py')

    webcam = launch_ros.actions.Node(
        package="v4l2_camera", executable="v4l2_camera_node",
        parameters=[
            {"image_size": [640,480]},
        ],
    )

    yolox_ros = launch_ros.actions.Node(
        package="yolox_ros_py", executable="yolox_ros",
        parameters=[
            {"image_size/width": 640},
            {"image_size/height": 480},
            {"yolo_type" : 'yolox-s'},
            {"fuse" : False},
            {"trt" : False},
            {"rank" : 0},
            {"ckpt_file" : yolox_ros_share_dir+"/yolox_s.pth"},
            {"conf" : 0.3},
            {"nmsthre" : 0.65},
            {"img_size" : 640},
        ],
    )

    rqt_graph = launch_ros.actions.Node(
        package="rqt_graph", executable="rqt_graph",
    )

    return launch.LaunchDescription([
        webcam,
        yolox_ros,
        rqt_graph
    ])