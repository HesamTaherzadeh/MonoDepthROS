from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    # RTAB-Map parameters
    parameters=[
        {
            'frame_id':'camera_link',
            'approx_sync':True,
            'publish_tf' : True
        },
        {
            'subscribe_depth': True,  # Subscribing to depth data for RGB-D SLAM
            'subscribe_rgb': True,    # Subscribing to RGB data
            'use_odometry': True,     # Use odometry information
            'subscribe_scan': False,  # For RGB-D only, no LIDAR
            'RGBD/ProximityBySpace': True  # For loop closure
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

        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', '/home/hesam/Desktop/datasets/kitti-odom/bag00.bag'],
            output='screen'
        ),

        ExecuteProcess(
            cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', "src/cfg/rviz.rviz"],
            output='screen'
        ),

        Node(
            package='rtabmap_odom', 
            executable='rgbd_odometry', 
            name="rgbd_odometry", 
            output="screen",
            parameters=[{
                "frame_id": "camera_link",                        # Default frame ID
                "odom_frame_id": "odom",                          # Default odom frame ID
                "publish_tf": True,                               # Publish TF between odom and base_link
                "ground_truth_frame_id": "",                      # Empty for no ground truth frame
                "ground_truth_base_frame_id": "",                 # Empty for no ground truth base frame
                "wait_for_transform": 0.2,                      # Default to not waiting for transform
                "approx_sync": True,            
                "expected_update_rate" : 10.0,                # Use approximate sync
                "approx_sync_max_interval": 0.0,                  # Maximum interval for approximate sync
                "config_path": "",                                # No custom config file path
                "topic_queue_size": 5,                           # Default queue size
                "sync_queue_size": 10,                            # Default sync queue size
                "qos": 1,                             # Default QoS for image
                "subscribe_rgbd": False,                           # Subscribe to RGBD input
                "guess_frame_id": "",                             # Default to not using guess frame
                "guess_min_translation": 0.0,                     # Default min translation for guess
                "guess_min_rotation": 0.0                         # Default min rotation for guess
            }],
            remappings=[
                ("rgb/image", "/left"),                           # Remap RGB image topic
                ("depth/image", "/depth_image"),                  # Remap depth image topic
                ("rgb/camera_info", "/camera2/left/camera_info"), # Remap camera info topi                
            ]        
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
            output='screen'
        )
    ])
