from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # Get the shared directory path for the 'depth_inference' package
    depth_inference_share_dir = get_package_share_directory('depth_inference')

    # Get the URDF file path
    urdf_file_path = os.path.join(depth_inference_share_dir, 'urdf', 'robot.urdf.xml')

    # Load URDF file content
    with open(urdf_file_path, 'r') as urdf_file:
        robot_description = urdf_file.read()

    # Define parameters for rtabmap_slam node
    parameters = [
        {
            'frame_id': 'odom',
            'approx_sync': True,
            'publish_tf': False
        },
        {
            'subscribe_depth': True,  # Subscribing to depth data for RGB-D SLAM
            'subscribe_rgb': True,    # Subscribing to RGB data
            'use_odometry': True,     # Use odometry information
            'subscribe_scan': False,  # For RGB-D only, no LIDAR
        }
    ]

    # Remap topics for RGB, depth, and camera info
    remappings = [
        ('/rgb/image', '/left'),
        ('/rgb/camera_info', '/camera2/left/camera_info'),
        ('/depth/image', '/depth_image')
    ]

    config_file_path = os.path.join(depth_inference_share_dir, 'cfg', 'config.yaml')
    rviz_file_path = os.path.join(depth_inference_share_dir, 'cfg', 'rviz.rviz')

    return LaunchDescription([
        # SLAM node
        Node(
            package='depth_inference',
            executable='slam_node',
            name='slam_node',
            output='screen',
            parameters=[config_file_path]  # Use the YAML config file
        ),

        # Robot state publisher node
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),

        # Rosbag play process
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', '/home/hesam/Desktop/datasets/kitti-odom/bag00.bag'],
            output='screen'
        ),

        # Rviz2 process
        ExecuteProcess(
            cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_file_path],
            output='screen'
        ),

        # RGB-D odometry node
        Node(
            package='rtabmap_odom', 
            executable='rgbd_odometry', 
            name="rgbd_odometry", 
            arguments=['--ros-args', '--log-level', 'error'],
            parameters=[{
                "frame_id": "base_link",
                "odom_frame_id": "odom",
                "publish_tf": True,
                "wait_for_transform": 0.2,
                "approx_sync": True,
                "approx_sync_max_interval": 0.0,
                "topic_queue_size": 100,
                "sync_queue_size": 100,
                "qos": 1,
                "subscribe_rgbd": False,
            }],
            remappings=[
                ("rgb/image", "/left"),
                ("depth/image", "/depth_image"),
                ("rgb/camera_info", "camera_info"),
            ],
            output='screen'
        ),

        # RTAB-Map SLAM node
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            parameters=parameters,
            arguments=['--ros-args', '--log-level', 'error'],
            remappings=remappings
        ),

        # # Static transform publisher: base to odom
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='base_to_odom_publisher',
        #     arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
        #     output='screen'
        # ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_publisher',
            arguments=['0', '0', '0', '0', '3.14', '0', 'map', 'odom'],
            output='screen'
        ),


        # Static transform publisher: map to odom
        
        # Static transform publisher: map to odom
        Node(
            package='depth_inference',
            executable='utils_node',
            name='utils_node',
            output='screen',
            parameters=[config_file_path] 
        )
        
    ])
