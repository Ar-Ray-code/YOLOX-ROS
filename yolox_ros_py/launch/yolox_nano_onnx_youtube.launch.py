import os
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_ros

def generate_launch_description():
    yolox_ros_share_dir = get_package_share_directory('yolox_ros_py')
    youtube_publisher_share_dir = get_package_share_directory('youtube_publisher')

    print("")
    print("===============================================================")
    print("Downloading video from YouTube...")
    print("Donwloading video takes about few minutes.")
    print("===============================================================")
    print("")
    time.sleep(3)

    youtube = launch_ros.actions.Node(
        package='youtube_publisher', executable='youtube_pub',
        parameters=[
            {'topic_name': '/image_raw'},
            {'cache_path': youtube_publisher_share_dir + '/cache'},
            {'video_url' : 'https://www.youtube.com/watch?v=CFLOiR2EbKM'},
            {'using_youtube_dl' : True},
            {'clear_cache_force' : False},
            {'width' : 640},
            {'height' : 360},
            {'imshow_is_show' : True}
        ],
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
    )

    return LaunchDescription([
        youtube,
        yolox_onnx,
    ])
