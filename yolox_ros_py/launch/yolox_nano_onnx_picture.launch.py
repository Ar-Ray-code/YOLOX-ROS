from matplotlib import image
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from urllib.request import urlretrieve
import os

def generate_launch_description():
    yolox_ros_share_dir = get_package_share_directory('yolox_ros_py')

    dog_path = os.path.join(yolox_ros_share_dir, "./", "dog.jpg")
    url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg"
    if not os.path.exists(dog_path):
        os.system("wget {} -O {}".format(url, dog_path))


    image_path = LaunchConfiguration('image_path', default=dog_path)
    image_path_arg = DeclareLaunchArgument(
        'image_path',
        default_value=dog_path,
        description='Image path'
    )

    image_pub = launch_ros.actions.Node(
        package='image_publisher',
        executable='image_publisher_node',
        name='image_publisher',
        arguments=[image_path],
    )
    yolox_onnx = launch_ros.actions.Node(
        package="yolox_ros_py", executable="yolox_onnx",output="screen",
        parameters=[
            {"input_shape/width": 416},
            {"input_shape/height": 416},

            {"with_p6" : False},
            {"model_path" : yolox_ros_share_dir+"/yolox_nano.onnx"},
            {"conf" : 0.3},
        ],
    )

    rqt_graph = launch_ros.actions.Node(
        package="rqt_graph", executable="rqt_graph",
    )

    return launch.LaunchDescription([
        image_path_arg,
        image_pub,
        yolox_onnx,
        # rqt_graph
    ])