#pragma once

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
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

// Kimera-VIO includes
#include "kimera-vio/utils/Timer.h"
#include "kimera-vio/utils/Statistics.h"
#include "kimera-vio/utils/UtilsOpenCV.h"
#include "kimera-vio/frontend/CameraParams.h"
#include "kimera-vio/dataprovider/DataProviderInterface.h"
#include "kimera-vio/dataprovider/MonoDataProviderModule.h"
#include "kimera-vio/imu-frontend/ImuFrontend-definitions.h"
#include "kimera-vio/pipeline/Pipeline.h"
#include "kimera-vio/pipeline/Pipeline-definitions.h"
#include "kimera-vio/pipeline/MonoImuPipeline.h"
#include "kimera-vio/pipeline/PipelineParams.h"

#include <models.hpp>
#include <context.hpp>

class SlamNode : public rclcpp::Node {
public:
    SlamNode();
    ~SlamNode();

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void map_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

    void save_cloud();

    // Publishers and Subscribers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_image_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscriber_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr car_base_odom_publisher_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Depth inference context
    std::shared_ptr<Context> context;

    // Storage for camera info
    sensor_msgs::msg::CameraInfo::SharedPtr last_camera_info_;

    // Cloud storage
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    int index;
    rclcpp::Time time;

    // Kimera-VIO pipeline
    VIO::VioParams vio_params_;
    std::unique_ptr<VIO::MonoImuPipeline> vio_pipeline_;
    VIO::CameraParams camera_params_;
    VIO::FrameId frame_id_{0};

    // Thread for pipeline spinning
    std::atomic<bool> pipeline_running_;
    std::thread pipeline_thread_;
};