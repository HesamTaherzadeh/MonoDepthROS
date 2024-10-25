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

    # Define RTAB-Map SLAM parameters
    rtabmap_params = [
        {
            'frame_id': 'odom',
            'approx_sync': True,
            'publish_tf': False
        },
        {
            'subscribe_depth': True,
            'subscribe_rgb': True,
            'subscribe_scan': False,
            'Mem/ImagePreDecimation': '1',
            'Mem/ImagePostDecimation': '1',
            'Grid/DepthDecimation': '1',
            'RGBD/NeighborLinkRefining': 'true',
            'kf/MaxDepth': 10,
            'kf/MinDepth': 0,
            'RGBD/CreateOccupancyGrid': 'true'
        }
    ]

    # Configuration file paths
    config_file_path = os.path.join(depth_inference_share_dir, 'cfg', 'config.yaml')
    rviz_file_path = os.path.join(depth_inference_share_dir, 'cfg', 'rviz.rviz')

    return LaunchDescription([
        # Depth inference SLAM node
        Node(
            package='depth_inference',
            executable='slam_node',
            name='slam_node',
            output='screen',
            parameters=[config_file_path]
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

        # RViz process
        ExecuteProcess(
            cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', rviz_file_path],
            output='screen'
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
            output='screen'
        ),

        # RTAB-Map SLAM node
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            arguments=['-d'],
            parameters=rtabmap_params,
            output='screen',
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
            output='screen'
        ),

        # Depth inference utility node
        Node(
            package='depth_inference',
            executable='utils_node',
            name='utils_node',
            output='screen',
            parameters=[config_file_path]
        )
    ])
