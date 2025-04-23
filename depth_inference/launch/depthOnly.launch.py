from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get shared directory for the 'depth_inference' package
    depth_inference_share_dir = get_package_share_directory('depth_inference')

    # URDF file path
    urdf_file_path = os.path.join(depth_inference_share_dir, 'urdf', 'robot.urdf.xml')
    with open(urdf_file_path, 'r') as urdf_file:
        robot_description = urdf_file.read()
        
    config_file_path = os.path.join(depth_inference_share_dir, 'cfg', 'config.yaml')
    rviz_file_path = os.path.join(depth_inference_share_dir, 'cfg', 'rviz.rviz')

    return LaunchDescription([
        Node(
            package='depth_inference',
            executable='slam_node',
            name='slam_node',
            output='screen',
            parameters=[config_file_path]
        ),

        # Robot state publisher node
        # Node(
        #     package='robot_state_publisher',
        #     executable='robot_state_publisher',
        #     output='log',
        #     parameters=[{'robot_description': robot_description}],
        # ),
        
        # Rosbag play process
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', '/home/hesam/Desktop/datasets/outputKITTIros2', '-r 0.8'],
            output='log'
        ),
        
        # RViz process
        ExecuteProcess(
            cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_file_path],
            output='log'
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_publisher',
            arguments=['0', '0', '0', '1.5708', '3.14159', '1.5708', 'map', 'odom'],
            output='log'
        ),

        # Depth inference utility node
        Node(
            package='depth_inference',
            executable='utils_node',
            name='utils_node',
            output='log',
            parameters=[config_file_path]
        ),
    ])
