from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import numpy as np

def generate_launch_description():

    urdf_file_path = 'src/depth_inference/urdf/robot.urdf.xml'

    with open(urdf_file_path, 'r') as urdf_file:
        robot_description = urdf_file.read()

    parameters=[
        {
            'frame_id':'camera_link',
            'approx_sync':True,
            'publish_tf' : False
        },
        {
            'subscribe_depth': True,  # Subscribing to depth data for RGB-D SLAM
            'subscribe_rgb': True,    # Subscribing to RGB data
            'use_odometry': True,     # Use odometry information
            'subscribe_scan': False,  # For RGB-D only, no LIDAR
        }
    ]

    # Remapping topics for RGB, depth, and camera info
    remappings=[
        ('/rgb/image', '/left'),
        ('/rgb/camera_info', '/camera2/left/camera_info'),
        ('/depth/image', '/depth_image')
    ]

    return LaunchDescription([
        Node(
            package='depth_inference',
            executable='slam_node',
            name='slam_node',
            output='screen',
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),


        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', '/home/hesam/Desktop/datasets/kitti-odom/bag00.bag'],
            output='screen'
        ),


        ExecuteProcess(
            cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', 'src/depth_inference/cfg/rviz.rviz'],
            output='screen'
        ),

        Node(
            package='rtabmap_odom', 
            executable='rgbd_odometry', 
            name="rgbd_odometry", 
            output="screen",
            parameters=[{
                "frame_id": "base_link",                        # Default frame ID
                "odom_frame_id": "odom",                          # Default odom frame ID
                "publish_tf": True,                               # Publish TF between odom and base_link
                "ground_truth_frame_id": "",                      # Empty for no ground truth frame
                "ground_truth_base_frame_id": "",                 # Empty for no ground truth base frame
                "wait_for_transform": 0.2,                      # Default to not waiting for transform
                "approx_sync": True,            
                "approx_sync_max_interval": 0.0,                  # Maximum interval for approximate sync
                "config_path": "",                                # No custom config file path
                "topic_queue_size": 100,                           # Default queue size
                "sync_queue_size": 100,                            # Default sync queue size
                "qos": 1,                             # Default QoS for image
                "subscribe_rgbd": False,                           # Subscribe to RGBD input
                "guess_frame_id": "",                             # Default to not using guess frame
                "guess_min_translation": 0.0,                     # Default min translation for guess
                "guess_min_rotation": 0.0
                # "initial_pose": "0 0 0 0 0 0"                         # Default min rotation for guess
            }],
            remappings=[
                ("rgb/image", "/left"),                           # Remap RGB image topic
                ("depth/image", "/depth_image"),                  # Remap depth image topic
                ("rgb/camera_info", "/camera2/left/camera_info"), # Remap camera info topi                
            ]        
        ),

        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=parameters,
            remappings=[
                ('/rgb/image', '/left'),
                ('/rgb/camera_info', '/camera2/left/camera_info'),
                ('/depth/image', '/depth_image'),
                # ('/odom', '/rgbd_odometry/odom')  # Remap odometry output from rgbd_odometry node
            ]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_odom_publisher',
            arguments=['0', '0', '0', '0', '0', '0' , 'odom', 'base_link'],
            output='screen'
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_publisher',
            arguments=['0', '0', '0', '3.14', '0', '-1.57' , 'odom', 'map'],
            output='screen'
        )
    ])
