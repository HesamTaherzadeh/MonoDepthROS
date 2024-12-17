#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

// Include your depth node header
#include "depth_node.hpp"
#include "yaets/tracing.hpp"


yaets::TraceSession session("session1.log");

SlamNode::SlamNode() : Node("slam_node"), cloud(std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>()) , vio_params_("/home/hesam/Desktop/Repos/Kimera-VIO/params/Euroc"){

    // Declare and get parameters
    this->declare_parameter("image_width", 196);
    this->declare_parameter("image_height", 616);
    this->declare_parameter("image_topic", "/camera2/left/image_raw");
    this->declare_parameter("model", "depth_pro");
    this->declare_parameter("onnx_model", "/home/hesam/Desktop/playground/depth_node/model/unidepthv2_vits14._big.onnx");
    this->declare_parameter("params_folder_path", "/home/hesam/Desktop/Repos/Kimera-VIO/params/Euroc"); // Adjust this path
    const std::string params_folder = this->get_parameter("params_folder_path").as_string();
    
    int width = this->get_parameter("image_width").as_int();
    int height = this->get_parameter("image_height").as_int();
    std::string onnx_model = this->get_parameter("onnx_model").as_string();
    std::string image_topic = this->get_parameter("image_topic").as_string();
    std::string model_name = this->get_parameter("model").as_string();

    RCLCPP_INFO(this->get_logger(), "Width : %d", width);
    RCLCPP_INFO(this->get_logger(), "Height : %d", height);
    RCLCPP_INFO(this->get_logger(), "image topic : %s", image_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "ONNX path : %s", onnx_model.c_str());
    RCLCPP_INFO(this->get_logger(), "Model : %s", model_name.c_str());
    RCLCPP_INFO(this->get_logger(), "Kimera params folder: %s", params_folder.c_str());

    index = 0;
    context = std::make_shared<Context>(modelTypeMap.at(model_name), onnx_model.c_str(), width, height);

    depth_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("depth_image", 100);
    left_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("left", 100);
    camera_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/rgb/camera_info", 100);

    image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic, 100,
        std::bind(&SlamNode::image_callback, this, std::placeholders::_1)
    );

    point_cloud_subscriber = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/cloud_map", 100,
        std::bind(&SlamNode::map_cloud_callback, this, std::placeholders::_1)
    );

    camera_info_subscriber_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/kitti/camera_color_left/camera_info", 100,
        std::bind(&SlamNode::camera_info_callback, this, std::placeholders::_1)
    );

    odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10,
        std::bind(&SlamNode::odom_callback, this, std::placeholders::_1)
    );

    car_base_odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/car/base/odom_corrected", 10);
    odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom_normalized", 10);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // IMU subscription (adjust topic as needed)
    imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data_raw", 200,
        std::bind(&SlamNode::imu_callback, this, std::placeholders::_1)
    );

    // Initialize Kimera-VIO parameters and pipeline
    vio_params_.frontend_type_ = VIO::FrontendType::kMonoImu; // Monocular pipeline

    vio_pipeline_ = std::make_unique<VIO::MonoImuPipeline>(vio_params_);
    vio_pipeline_->registerShutdownCallback([this]() {
        RCLCPP_INFO(this->get_logger(), "Kimera-VIO pipeline shutting down");
    });

    // Start Kimera-VIO pipeline spin in a separate thread
    pipeline_running_ = true;
    pipeline_thread_ = std::thread([this]() {
        while (rclcpp::ok() && pipeline_running_) {
            if (!vio_pipeline_->spin()) {
                pipeline_running_ = false;
            }
        }
        vio_pipeline_->shutdown();
    });
}

SlamNode::~SlamNode() {
    RCLCPP_INFO(this->get_logger(), "SlamNode is shutting down, saving point cloud...");
    pipeline_running_ = false;
    if (pipeline_thread_.joinable()) {
        pipeline_thread_.join();
    }
    this->save_cloud();
}

void SlamNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    TRACE_EVENT(session);
    std::cout << "recieved the image" <<std::endl;
    cv::Mat input_image = cv_bridge::toCvShare(msg, "bgr8")->image;

    if (input_image.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Received an empty image!");
        return;
    }

    std_msgs::msg::Header header = msg->header;
    header.frame_id = "left"; 

    // Publish original left image
    sensor_msgs::msg::Image::SharedPtr left_msg = cv_bridge::CvImage(
        header, 
        sensor_msgs::image_encodings::BGR8, 
        input_image
    ).toImageMsg();
    left_image_publisher_->publish(*left_msg);

    // Run depth inference
    cv::Mat depth_image = context->runInference(input_image);

    double minVal, maxVal;
    cv::minMaxLoc(depth_image, &minVal, &maxVal);
    std::cout << "Depth Image Min: " << minVal << ", Max: " << maxVal << std::endl;

    header.frame_id = "camera_link"; 
    sensor_msgs::msg::Image::SharedPtr depth_msg = cv_bridge::CvImage(
        header, 
        sensor_msgs::image_encodings::TYPE_32FC1, 
        depth_image
    ).toImageMsg();
    depth_image_publisher_->publish(*depth_msg);

    if (last_camera_info_) {
        sensor_msgs::msg::CameraInfo camera_info_msg = *last_camera_info_;
        camera_info_msg.header = header;
        camera_info_msg.height = input_image.rows;
        camera_info_msg.width = input_image.cols;
        camera_info_publisher_->publish(camera_info_msg);
    }

    if (!camera_params_.intrinsics_.empty()) {
        VIO::Timestamp timestamp = 
            (VIO::Timestamp)(msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec);

        // Create a Frame for Kimera-VIO
        VIO::Frame::UniquePtr frame = std::make_unique<VIO::Frame>(frame_id_++, timestamp, camera_params_, input_image);
        // Feed frame into Kimera-VIO pipeline
        vio_pipeline_->fillLeftFrameQueue(std::move(frame));
    }

    ++index;
}
void SlamNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  last_camera_info_ = msg;

  // Extract intrinsics from msg->k
  double fx = msg->k[0];
  double fy = msg->k[4];
  double cx = msg->k[2];
  double cy = msg->k[5];
  double s = 0.0; // Usually no skew

  // Set calibration (assuming a pinhole + radial-tangential model)
  camera_params_.K_ = (cv::Mat_<double>(3, 3) << 
                     fx, s,  cx,
                     0,  fy, cy,
                     0,  0,  1);
  // Map ROS distortion model to Kimera-VIO DistortionModel
  // For "plumb_bob" in ROS, use DistortionModel::RADTAN
  if (msg->distortion_model == "plumb_bob" && msg->d.size() >= 5) {
    camera_params_.distortion_model_ = VIO::DistortionModel::RADTAN;
    // Distortion: k1, k2, t1(p1), t2(p2), k3
    camera_params_.distortion_coeff_ = {msg->d[0], msg->d[1], msg->d[2], msg->d[3], msg->d[4]};
  } else {
    // If distortion model is unknown or doesn't match Kimera's supported ones:
    camera_params_.distortion_model_ = VIO::DistortionModel::NONE;
    camera_params_.distortion_coeff_.clear();
  }

}

void SlamNode::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    TRACE_EVENT(session);

    auto corrected_msg = *msg;
    tf2::Quaternion q1;
    q1.setRPY(3.14, 0, -1.57);

    tf2::Quaternion q(
        corrected_msg.pose.pose.orientation.x,
        corrected_msg.pose.pose.orientation.y,
        corrected_msg.pose.pose.orientation.z,
        corrected_msg.pose.pose.orientation.w
    );
    q = q * q1;
    q.normalize();

    corrected_msg.pose.pose.orientation = tf2::toMsg(q);
    odom_publisher_->publish(corrected_msg);
}

void SlamNode::map_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!cloud) {
        RCLCPP_ERROR(this->get_logger(), "Cloud is not initialized!");
        return;
    }

    pcl::fromROSMsg(*msg, *cloud);
    RCLCPP_INFO(this->get_logger(), "PointCloud received and added to cloud.");
}

void SlamNode::imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg) {
    VIO::Timestamp imu_timestamp = 
        (VIO::Timestamp)(imu_msg->header.stamp.sec * 1e9 + imu_msg->header.stamp.nanosec);

    VIO::ImuAccGyr imu_data;
    imu_data << imu_msg->linear_acceleration.x,
                imu_msg->linear_acceleration.y,
                imu_msg->linear_acceleration.z,
                imu_msg->angular_velocity.x,
                imu_msg->angular_velocity.y,
                imu_msg->angular_velocity.z;

    VIO::ImuMeasurement imu_measurement(imu_timestamp, imu_data);
    vio_pipeline_->fillSingleImuQueue(imu_measurement);
}

void SlamNode::save_cloud() {
    TRACE_EVENT(session);
    if (cloud && !cloud->empty()) {
        pcl::io::savePCDFileASCII("/home/hesam/Desktop/pointcloud.pcd", *cloud);
        RCLCPP_INFO(this->get_logger(), "PointCloud saved to /home/hesam/Desktop/pointcloud.pcd.");
    } else {
        RCLCPP_WARN(this->get_logger(), "No point cloud data to save.");
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
