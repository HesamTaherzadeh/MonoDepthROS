#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry


class RGBDOdometryNode(Node):
    def __init__(self):
        super().__init__('rgbd_odometry_node')

        # Parameters
        self.declare_parameter('rgb_topic', '/kitti/camera_color_left/image_raw')
        self.declare_parameter('depth_topic', 'depth_image')
        self.declare_parameter('camera_intrinsics', [7.188560000000e+02, 7.188560000000e+02, 6.071928000000e+02,1.852157000000e+02])  # fx, fy, cx, cy

        # Load parameters
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_intrinsics = self.get_parameter('camera_intrinsics').value

        # Camera intrinsic matrix
        self.K = np.array([[self.camera_intrinsics[0], 0, self.camera_intrinsics[2]],
                           [0, self.camera_intrinsics[1], self.camera_intrinsics[3]],
                           [0, 0, 1]], dtype=np.float32)

        # Subscribers
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)

        # Pose Publisher
        self.odom_pub = self.create_publisher(Odometry, '/rgbd_odometry/pose', 10)

        # Other variables
        self.bridge = CvBridge()
        self.prev_rgb = None
        self.prev_depth = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_pose = np.eye(4)  # Initial pose (identity matrix)
        self.detector = cv2.ORB_create()  # ORB detector for feature extraction
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Brute-force matcher

    def rgb_callback(self, msg):
        """Callback for RGB images."""
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        """Callback for depth images."""
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1') 


        if hasattr(self, 'rgb_image'):
            # Process RGB and depth together
            pose = self.process_frame(self.rgb_image, depth_image)
            self.publish_odom(pose)

    def process_frame(self, rgb_image, depth_image):
        """
        Process a new RGB-D frame to compute odometry.
        """
        # Step 1: Extract keypoints and descriptors
        keypoints, descriptors = self.detector.detectAndCompute(rgb_image, None)

        if self.prev_rgb is None:
            # Initialize with the first frame
            self.prev_rgb = rgb_image
            self.prev_depth = depth_image
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return self.prev_pose

        # Step 2: Match features between the current and previous frames
        matches = self.matcher.match(descriptors, self.prev_descriptors)

        # Step 3: Filter matches to remove outliers
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.75)]  # Keep the top 75% matches

        # Step 4: Get 2D-3D correspondences
        src_points, dst_points, src_3d_points = self.get_correspondences(good_matches, keypoints, self.prev_keypoints, depth_image)
        src_3d_points = np.asarray(src_3d_points, dtype=np.float64).reshape(-1, 3)
        src_points = np.asarray(src_points, dtype=np.float64).reshape(-1, 2)

        # Step 5: Estimate pose using PnP
        if len(src_3d_points) >= 4:  # Minimum points needed for PnP
            success, rvec, tvec = cv2.solvePnP(src_3d_points, src_points, self.K, None)

            if success:
                # Convert rotation vector to matrix
                R_mat = cv2.Rodrigues(rvec)[0]

                # Build the transformation matrix
                T = np.eye(4, dtype=float)
                T[:3, :3] = R_mat
                T[:3, 3] = tvec.ravel()

                # Update the global pose
                self.prev_pose =  self.prev_pose @ T

        # Update previous frame data
        self.prev_rgb = rgb_image
        self.prev_depth = depth_image
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        return self.prev_pose

    # In get_correspondences method:
    def get_correspondences(self, matches, keypoints, prev_keypoints, depth_image):
        src_points = []
        dst_points = []
        src_3d_points = []

        for match in matches:
            kp_curr = keypoints[match.queryIdx].pt
            kp_prev = prev_keypoints[match.trainIdx].pt

            u, v = int(kp_curr[0]), int(kp_curr[1])
            
            # Use depth from previous frame instead of current frame
            depth = self.prev_depth[v, u]
            if depth > 0:
                x = (u - self.K[0, 2]) * depth / self.K[0, 0]
                y = (v - self.K[1, 2]) * depth / self.K[1, 1]
                z = depth

                src_points.append(kp_curr)
                dst_points.append(kp_prev)
                src_3d_points.append([x, y, z])

        return (
            np.array(src_points, dtype=np.float32),
            np.array(dst_points, dtype=np.float32),
            np.array(src_3d_points, dtype=np.float32),
        )


    def publish_odom(self, pose_matrix):
        """
        Publish the current pose as an Odometry message.
        """
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'map'  # Frame of reference for the pose
        odom_msg.child_frame_id = 'base_link'  # Frame of the moving robot (child link)

        # Extract translation (position)
        odom_msg.pose.pose.position.x = pose_matrix[0, 3]
        odom_msg.pose.pose.position.y = pose_matrix[1, 3]
        odom_msg.pose.pose.position.z = pose_matrix[2, 3]

        # Extract rotation (convert from rotation matrix to quaternion)
        rot = pose_matrix[:3, :3]
        q = R.from_matrix(rot).as_quat()
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        # Set covariance (optional, set to zero if no uncertainty is calculated)
        odom_msg.pose.covariance = [0.0] * 36  # 6x6 covariance matrix for pose

        # Linear and angular velocity (optional, assumed zero in this example)
        odom_msg.twist.twist.linear.x = 0.0
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = 0.0

        # Twist covariance (optional)
        odom_msg.twist.covariance = [0.0] * 36  # 6x6 covariance matrix for twist

        # Publish the odometry message
        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RGBDOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
