#include "depth_node.hpp"

SlamNode::SlamNode() : Node("slam_node") 
, cloud(std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>()){
    /**
    !TODO : 
        - Resolve the issue about odometry failure 
     */

    this->declare_parameter ("image_width", 196);  
    this->declare_parameter("image_height", 616); 
    this->declare_parameter("image_topic", "/camera2/left/image_raw"); 
    this->declare_parameter("onnx_model", "/home/hesam/Desktop/playground/depth_node/model/unidepthv2_vits14_simp.onnx"); 


    // Retrieve parameters
    int width = this->get_parameter("image_width").as_int();
    int height = this->get_parameter("image_height").as_int();
    std::string onnx_model = this->get_parameter("onnx_model").as_string();
    std::string image_topic = this->get_parameter("image_topic").as_string();

    RCLCPP_INFO(this->get_logger(), "Width : %d", width);
    RCLCPP_INFO(this->get_logger(), "Height : %d", height);
    RCLCPP_INFO(this->get_logger(), "image topic : %s", image_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "ONNX path : %s", onnx_model.c_str());

    model_runner_ = std::make_shared<ModelRunner>(onnx_model.c_str(), width, height);

    depth_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("depth_image", 100);
    left_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("left", 100);
    camera_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", 100);

    image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic, 100,
        std::bind(&SlamNode::image_callback, this, std::placeholders::_1)
    );


    point_cloud_subscriber = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/cloud_map", 100,
        std::bind(&SlamNode::map_cloud_callback, this, std::placeholders::_1)
    );

    camera_info_subscriber_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera2/left/camera_info", 100,
        std::bind(&SlamNode::camera_info_callback, this, std::placeholders::_1)
    );

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    car_base_odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/car/base/odom", 10,
        std::bind(&SlamNode::car_base_odom_callback, this, std::placeholders::_1)
    );

    odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10,
        std::bind(&SlamNode::odom_callback, this, std::placeholders::_1)
    );

    car_base_odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/car/base/odom_corrected", 10);
    odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom_normalized", 10);

    // std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> cloud = std::make_shared<pcl::PointXYZRGB>();
}

 SlamNode::~SlamNode() {
    RCLCPP_INFO(this->get_logger(), "SlamNode is shutting down, saving point cloud...");
    this->save_cloud();
    };

void SlamNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv::Mat input_image = cv_bridge::toCvShare(msg)->image;

    if (input_image.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Received an empty image!");
        return;
    }

    std_msgs::msg::Header header;
    header.stamp = this->now();
    header.frame_id = "left"; 

    sensor_msgs::msg::Image::SharedPtr left_msg = cv_bridge::CvImage(
        header, 
        sensor_msgs::image_encodings::BGR8, 
        input_image
    ).toImageMsg();
    left_image_publisher_->publish(*left_msg);

    cv::Mat depth_image = model_runner_->runInference(input_image);

    // cv::Mat depth_converted;
    // depth_image.convertTo(depth_converted, CV_8UC1, 1);

    #ifdef OPENCV_IMSHOW
        cv::imshow("depth", depth_image);
        cv::waitKey(1);
    #endif

    sensor_msgs::msg::Image::SharedPtr depth_msg = cv_bridge::CvImage(
        header, 
        sensor_msgs::image_encodings::TYPE_32FC1, 
        depth_image
    ).toImageMsg();
    depth_image_publisher_->publish(*depth_msg);

    if (last_camera_info_) {
        sensor_msgs::msg::CameraInfo camera_info_msg = *last_camera_info_;
        camera_info_msg.header.stamp = header.stamp;
        camera_info_publisher_->publish(camera_info_msg);
    }

}

void SlamNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    last_camera_info_ = msg;
}

void SlamNode::car_base_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    auto corrected_msg = *msg;
    corrected_msg.header.stamp = this->now();
    car_base_odom_publisher_->publish(corrected_msg);
}

void SlamNode::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    auto corrected_msg = *msg;

    tf2::Quaternion q1;
    q1.setRPY(3.14, 0, -1.57);

    tf2::Quaternion q(
        corrected_msg.pose.pose.orientation.w,
        corrected_msg.pose.pose.orientation.x,
        corrected_msg.pose.pose.orientation.y,
        corrected_msg.pose.pose.orientation.z
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

void SlamNode::save_cloud() {
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

    return 0;
}
