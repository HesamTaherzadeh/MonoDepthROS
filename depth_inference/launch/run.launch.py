from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
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
        # Depth inference SLAM node
        Node(
            package='depth_inference',
            executable='slam_node',
            name='slam_node',
            output='log',
            parameters=[config_file_path]
        ),

        # Robot state publisher node
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='log',
            parameters=[{'robot_description': robot_description}],
        ),

        # Rosbag play process
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', '/home/hesam/Desktop/datasets/kitti-odom/bag00.bag'],
            output='log'
        ),

        # RViz process
        ExecuteProcess(
            cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_file_path],
            output='log'
        ),

        # RGB-D odometry node
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='rgbd_odometry',
            arguments=['--ros-args', '--log-level', 'error'],
            parameters=[{
                "frame_id": "base_link",
                "publish_tf": True,
                "wait_for_transform": 0.2,
                "approx_sync": True,
                "approx_sync_max_interval": 0.0,
                "topic_queue_size": 100000000,
                "sync_queue_size": 1000000000,
                "qos": 1,
                "subscribe_rgbd": False
            }],
            remappings=[
                ("rgb/image", "/left"),
                ("depth/image", "/depth_image"),
                ("rgb/camera_info", "camera_info")
            ],
            output='log'
        ),

        # RTAB-Map SLAM node
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            arguments=['--delete_db_on_start', 'udebug'],
            output='screen',
            parameters=[{
                'Mem/ImagePreDecimation': '1',          # No pre-decimation on input images
                'Mem/ImagePostDecimation': '1',         # No post-decimation on input images
                'Grid/RangeMax': '0',                   # No max range cutoff, so max range of sensor is used
                'Grid/RangeMin': '0.1',                 # Minimal range for grid mapping
                'Grid/DepthDecimation': '1',            # No decimation on depth images
                'Grid/CellSize': '0.02',                # Small cell size for high-resolution occupancy grid (2 cm)
                'Grid/FromDepth': 'true',               # Use depth information for the occupancy grid
                'RGBD/LinearUpdate': '0',               # Update the map even if the robot moved only a bit
                'RGBD/AngularUpdate': '0',              # Update the map even if the robot rotated only a bit
                'RGBD/ProximityPathMaxNeighbors': '0',  # Use all neighbors to create a dense map
                'RGBD/NeighborLinkRefining': 'true',    # Refine links to neighbors for dense mapping
                'Optimizer/GravitySigma': '0',          # Disable gravity constraint to allow more freedom in optimization
                'RGBD/OptimizeFromGraphEnd': 'true',    # Optimize from the last pose to refine the map
            }],
            remappings=[
                ('/rgb/image', '/left'),
                ('/rgb/camera_info', 'camera_info'),
                ('/depth/image', '/depth_image')
            ]
        ),

        # Static transform publisher node (map to odom)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
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
