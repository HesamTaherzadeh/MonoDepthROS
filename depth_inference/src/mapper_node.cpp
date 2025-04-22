#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp> // For converting geometry_msgs to Eigen
#include <Eigen/Geometry>

// PCL specific includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

#include <mutex>
#include <fstream> // For file checking/saving if needed beyond PCL

// Define the point type we'll use
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;

class DenseMapperNode : public rclcpp::Node {
public:
    DenseMapperNode()
        : Node("dense_mapper_node"),
          camera_info_received_(false)
    {
        RCLCPP_INFO(this->get_logger(), "Initializing DenseMapperNode...");

        // --- Parameters ---
        this->declare_parameter<std::string>("rgb_image_topic", "left");
        this->declare_parameter<std::string>("depth_image_topic", "depth_image");
        this->declare_parameter<std::string>("odom_topic", "/orbslam3/odom");
        this->declare_parameter<std::string>("camera_info_topic", "/kitti/camera_color_left/camera_info");
        this->declare_parameter<double>("filter_distance_m", 10.0);
        this->declare_parameter<double>("voxel_leaf_size_m", 0.2);
        this->declare_parameter<std::string>("map_save_path", "/home/hesam/Desktop/Kitti_dense_map.pcd");
        this->declare_parameter<int>("sync_queue_size", 10);

        rgb_topic_ = this->get_parameter("rgb_image_topic").as_string();
        depth_topic_ = this->get_parameter("depth_image_topic").as_string();
        odom_topic_ = this->get_parameter("odom_topic").as_string();
        camera_info_topic_ = this->get_parameter("camera_info_topic").as_string();
        filter_distance_ = this->get_parameter("filter_distance_m").as_double();
        voxel_leaf_size_ = this->get_parameter("voxel_leaf_size_m").as_double();
        map_save_path_ = this->get_parameter("map_save_path").as_string();
        sync_queue_size_ = this->get_parameter("sync_queue_size").as_int();


        RCLCPP_INFO(this->get_logger(), "Parameters:");
        RCLCPP_INFO(this->get_logger(), "  RGB Topic: %s", rgb_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Depth Topic: %s", depth_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Odometry Topic: %s", odom_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Camera Info Topic: %s", camera_info_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Filter Distance (m): %.2f", filter_distance_);
        RCLCPP_INFO(this->get_logger(), "  Voxel Leaf Size (m): %.2f", voxel_leaf_size_);
        RCLCPP_INFO(this->get_logger(), "  Map Save Path: %s", map_save_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Sync Queue Size: %d", sync_queue_size_);


        // --- Initialization ---
        map_cloud_.reset(new PointCloudT());
        voxel_filter_.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);

        // --- Subscribers ---
        // Use message_filters subscribers for synchronization
        rgb_sub_.subscribe(this, rgb_topic_);
        depth_sub_.subscribe(this, depth_topic_);
        odom_sub_.subscribe(this, odom_topic_);

        // Regular subscriber for camera info (often published latched)
        // Use a reliable QoS profile for camera info
        auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/kitti/camera_color_left/camera_info", 100 , std::bind(&DenseMapperNode::cameraInfoCallback, this, _1));

        // --- Synchronizer ---
        // Sync Policy: ApproximateTime for RGB, Depth, and Odometry
        sync_ = std::make_shared<Sync>(SyncPolicy(10), rgb_sub_, depth_sub_, odom_sub_);
        sync_->setMaxIntervalDuration(rclcpp::Duration(0, 100000000));
        sync_->registerCallback(std::bind(&DenseMapperNode::jointCallback, this, _1, _2, _3));

        RCLCPP_INFO(this->get_logger(), "DenseMapperNode initialized. Waiting for camera info and messages...");
    }

    // --- Destructor: Save the map ---
    ~DenseMapperNode() {
        RCLCPP_INFO(this->get_logger(), "Node shutting down. Saving final map...");
        if (!map_cloud_ || map_cloud_->empty()) {
            RCLCPP_WARN(this->get_logger(), "Map is empty, nothing to save.");
            return;
        }

        try {
            // Apply final downsampling before saving
            PointCloudT::Ptr final_filtered_cloud(new PointCloudT());
            { // Scope for lock guard
                std::lock_guard<std::mutex> lock(map_mutex_);
                if (!map_cloud_->empty()){ // Double check after locking
                    voxel_filter_.setInputCloud(map_cloud_);
                    voxel_filter_.filter(*final_filtered_cloud);
                } else {
                    RCLCPP_WARN(this->get_logger(), "Map became empty before saving. Nothing to save.");
                    return;
                 }

            } // Mutex released here

            if (final_filtered_cloud->empty()){
                 RCLCPP_WARN(this->get_logger(), "Map is empty after final downsampling. Nothing to save.");
                 return;
            }

            RCLCPP_INFO(this->get_logger(), "Final map has %ld points after downsampling. Saving to %s",
                       final_filtered_cloud->size(), map_save_path_.c_str());

            // Save the final filtered cloud
            if (pcl::io::savePCDFileASCII(map_save_path_, *final_filtered_cloud) == 0) { 
                RCLCPP_INFO(this->get_logger(), "Map saved successfully to %s", map_save_path_.c_str());
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to save map to %s", map_save_path_.c_str());
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception during map saving: %s", e.what());
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "Unknown exception during map saving.");
        }
    }


private:
    // --- Typedefs for message_filters ---
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image, nav_msgs::msg::Odometry> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Sync;

    // --- Member Variables ---
    // Parameters
    std::string rgb_topic_;
    std::string depth_topic_;
    std::string odom_topic_;
    std::string camera_info_topic_;
    double filter_distance_;
    double voxel_leaf_size_;
    std::string map_save_path_;
    int sync_queue_size_;


    // ROS Subscriptions & Synchronization
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    std::shared_ptr<SyncPolicy> sync_policy_;
    std::shared_ptr<Sync> sync_;

    // PCL Map Data & Processing
    PointCloudT::Ptr map_cloud_;
    pcl::VoxelGrid<PointT> voxel_filter_;
    std::mutex map_mutex_; // To protect access to map_cloud_

    // Camera Intrinsics
    bool camera_info_received_;
    double fx_, fy_, cx_, cy_;
    std::mutex camera_info_mutex_; // Protect access during callback


    // --- Callbacks ---

    // Callback to store camera intrinsics
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
         if (camera_info_received_) {
            return; // Already received
         }
        std::lock_guard<std::mutex> lock(camera_info_mutex_);
        fx_ = msg->k[0]; // Projection matrix K: [fx 0 cx; 0 fy cy; 0 0 1]
        fy_ = msg->k[4];
        cx_ = msg->k[2];
        cy_ = msg->k[5];
        camera_info_received_ = true;
        RCLCPP_INFO(this->get_logger(), "Camera info received: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", fx_, fy_, cx_, cy_);

        // Unsubscribe after receiving info if it's static
        // camera_info_sub_.reset(); // Optional: uncomment if info never changes
    }


    // Joint callback for synchronized RGB, Depth, and Odometry messages
    void jointCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
        const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg)
    {
        // Check if camera info is available
        { // Scope for camera_info_mutex lock
            std::lock_guard<std::mutex> lock(camera_info_mutex_);
            if (!camera_info_received_) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for camera info...");
                return;
            }
            // Copy intrinsics locally to avoid holding lock during processing
             double fx = fx_;
             double fy = fy_;
             double cx = cx_;
             double cy = cy_;
        } // camera_info_mutex released

        try {
            // --- Convert ROS messages ---
            // Convert RGB image
            cv_bridge::CvImagePtr cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
            // Convert Depth image - ensure it's 32FC1 (meters)
            cv_bridge::CvImagePtr cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);

            const cv::Mat& rgb_image = cv_rgb_ptr->image;
            const cv::Mat& depth_image = cv_depth_ptr->image;

            // --- Get Odometry Pose ---
            Eigen::Isometry3d odom_pose_eigen;
            tf2::fromMsg(odom_msg->pose.pose, odom_pose_eigen);
            // Extract position for distance check later
            Eigen::Vector3d odom_position = odom_pose_eigen.translation();

            // --- Create Point Cloud from current frame ---
            PointCloudT::Ptr current_scan(new PointCloudT());
            current_scan->header.frame_id = odom_msg->header.frame_id; // Map frame
            pcl_conversions::toPCL(odom_msg->header.stamp, current_scan->header.stamp); // Use PCL timestamp format (microseconds)
            current_scan->is_dense = false; // Can contain NaN points initially
            current_scan->height = 1; // Unordered point cloud

            // Iterate through the depth image pixels
            for (int v = 0; v < depth_image.rows; ++v) {
                for (int u = 0; u < depth_image.cols; ++u) {
                    float depth = depth_image.at<float>(v, u);

                    // Check for valid depth (not NaN and positive)
                    if (std::isnan(depth) || depth <= 0.0) {
                        continue;
                    }

                    // --- Project pixel to 3D point in camera frame ---
                    Eigen::Vector3d point_camera;
                    point_camera.z() = depth;
                    point_camera.x() = (u - cx_) * depth / fx_;
                    point_camera.y() = (v - cy_) * depth / fy_;

                    // --- Transform point to map frame using Odometry ---
                    Eigen::Vector3d point_map = odom_pose_eigen * point_camera;

                    // --- Filter based on distance from current odometry position ---
                    double distance_to_sensor = (point_map - odom_position).norm();

                    if (distance_to_sensor >= filter_distance_) {
                        // Get color from RGB image (OpenCV is BGR)
                        cv::Vec3b color = rgb_image.at<cv::Vec3b>(v, u);

                        // Create PCL point
                        PointT pcl_point;
                        pcl_point.x = static_cast<float>(point_map.x());
                        pcl_point.y = static_cast<float>(point_map.y());
                        pcl_point.z = static_cast<float>(point_map.z());
                        pcl_point.b = color[0];
                        pcl_point.g = color[1];
                        pcl_point.r = color[2];

                        current_scan->points.push_back(pcl_point);
                    }
                }
            }
            current_scan->width = current_scan->points.size(); // Update width

            if (current_scan->points.empty()) {
                 return; // No valid points added in this scan
            }

            // --- Add current scan to global map and downsample ---
            { // Scope for map_mutex lock
                std::lock_guard<std::mutex> lock(map_mutex_);

                // Add new points
                *map_cloud_ += *current_scan;

                // Downsample the combined cloud
                PointCloudT::Ptr temp_cloud(new PointCloudT());
                voxel_filter_.setInputCloud(map_cloud_);
                voxel_filter_.filter(*temp_cloud);
                map_cloud_.swap(temp_cloud); // Efficiently replace map with filtered version

            } // map_mutex released

            RCLCPP_DEBUG(this->get_logger(), "Processed scan. Map points: %ld", map_cloud_->size());

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        } catch (const tf2::TransformException &ex) {
             RCLCPP_ERROR(this->get_logger(), "TF Exception: %s", ex.what());
        } catch (const std::exception& e) {
             RCLCPP_ERROR(this->get_logger(), "Standard exception in callback: %s", e.what());
        } catch (...) {
             RCLCPP_ERROR(this->get_logger(), "Unknown exception in callback.");
        }
    }
}; // End class DenseMapperNode

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DenseMapperNode>();
    rclcpp::spin(node);
    // Destructor will be called upon shutdown, saving the map.
    rclcpp::shutdown();
    return 0;
}