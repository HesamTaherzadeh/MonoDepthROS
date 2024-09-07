#pragma once

#include "model_reader.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include <opencv2/opencv.hpp>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h" 

class SlamNode : public rclcpp::Node {
public:
    SlamNode();

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void car_base_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);

    std::shared_ptr<ModelRunner> model_runner_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_image_publisher_;  
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_publisher_;  
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscriber_;  
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr car_base_odom_subscriber_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr car_base_odom_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    sensor_msgs::msg::CameraInfo::SharedPtr last_camera_info_;
    rclcpp::Time time;
};

