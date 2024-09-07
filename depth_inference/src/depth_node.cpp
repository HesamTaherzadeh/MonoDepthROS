#include "depth_node.hpp"

SlamNode::SlamNode() : Node("slam_node") {
    model_runner_ = std::make_shared<ModelRunner>("/home/hesam/Desktop/playground/depth_node/src/unidepthv2_vits14_simp.onnx", 644, 364);

    depth_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("depth_image", 100);
    left_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/left", 100);
    camera_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/camera2/left/camera_info", 100);

    image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera2/left/image_raw", 100,
        std::bind(&SlamNode::image_callback, this, std::placeholders::_1)
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

}

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

    cv::Mat depth_converted;
    depth_image.convertTo(depth_converted, CV_16UC1, 1000.0);

    sensor_msgs::msg::Image::SharedPtr depth_msg = cv_bridge::CvImage(
        header, 
        sensor_msgs::image_encodings::TYPE_16UC1, 
        depth_converted
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

    double temp_x = corrected_msg.pose.pose.position.x;
    corrected_msg.pose.pose.position.x = -corrected_msg.pose.pose.position.z;
    corrected_msg.pose.pose.position.z = -corrected_msg.pose.pose.position.y;
    corrected_msg.pose.pose.position.y = temp_x;

    tf2::Quaternion q(
        corrected_msg.pose.pose.orientation.w,
        corrected_msg.pose.pose.orientation.x,
        corrected_msg.pose.pose.orientation.y,
        corrected_msg.pose.pose.orientation.z
    );
    q.normalize();

    corrected_msg.pose.pose.orientation = tf2::toMsg(q);
    odom_publisher_->publish(corrected_msg);
}
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SlamNode>());
    rclcpp::shutdown();
    return 0;
}
