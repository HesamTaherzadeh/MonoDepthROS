#include "model_reader.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include <opencv2/opencv.hpp>

class SlamNode : public rclcpp::Node {
public:
    SlamNode() : Node("slam_node") {
        model_runner_ = std::make_shared<ModelRunner>("/home/hesam/Desktop/playground/depth_node/src/unidepthv2_vits14_simp.onnx", 644, 364);

        depth_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("depth_image", 20);
        left_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/left", 20);
        camera_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/camera2/left/camera_info", 20);

        image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera2/left/image_raw", 20,
            std::bind(&SlamNode::image_callback, this, std::placeholders::_1)
        );

        camera_info_subscriber_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera2/left/camera_info", 20,
            std::bind(&SlamNode::camera_info_callback, this, std::placeholders::_1)
        );

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        publish_base_link_transform();
        publish_left_transform();
    }

private:
    // Callback for image
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        // Convert ROS image message to OpenCV format
        cv::Mat input_image = cv_bridge::toCvShare(msg)->image;

        if (input_image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Received an empty image!");
            return;
        }

        // Republish the left image with the same timestamp
        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = "left"; 

        sensor_msgs::msg::Image::SharedPtr left_msg = cv_bridge::CvImage(
            header, 
            sensor_msgs::image_encodings::BGR8, 
            input_image
        ).toImageMsg();
        left_image_publisher_->publish(*left_msg);

        // Run inference on the image to get the depth image
        cv::Mat depth_image = model_runner_->runInference(input_image);

        // Convert depth image to 16UC1 (16-bit unsigned integer) format
        cv::Mat depth_converted;
        depth_image.convertTo(depth_converted, CV_16UC1, 1000.0);  // Scale to meters

        // Publish depth image with the same timestamp
        sensor_msgs::msg::Image::SharedPtr depth_msg = cv_bridge::CvImage(
            header, 
            sensor_msgs::image_encodings::TYPE_16UC1, 
            depth_converted
        ).toImageMsg();
        depth_image_publisher_->publish(*depth_msg);

        // Update and republish the camera info with the same timestamp
        if (last_camera_info_) {
            sensor_msgs::msg::CameraInfo camera_info_msg = *last_camera_info_;
            camera_info_msg.header.stamp = header.stamp;
            camera_info_publisher_->publish(camera_info_msg);
        }

        // Publish transforms
        publish_base_link_transform();
        publish_left_transform();
    }

    // Callback for camera info
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        last_camera_info_ = msg;  // Cache the latest camera info message
    }

    void publish_base_link_transform() {
        geometry_msgs::msg::TransformStamped transformStamped;

        transformStamped.header.stamp = this->now();
        transformStamped.header.frame_id = "base_link";
        transformStamped.child_frame_id = "camera_link";  // Set the child frame to /left

        transformStamped.transform.translation.x = 0.0;
        transformStamped.transform.translation.y = 0.0;
        transformStamped.transform.translation.z = 0.0;

        transformStamped.transform.rotation.x = 0.0;
        transformStamped.transform.rotation.y = 0.0;
        transformStamped.transform.rotation.z = 0.0;
        transformStamped.transform.rotation.w = 1.0;

        tf_broadcaster_->sendTransform(transformStamped);
    }

    void publish_left_transform() {
        geometry_msgs::msg::TransformStamped left_transformStamped;

        left_transformStamped.header.stamp = this->now();
        left_transformStamped.header.frame_id = "camera_link"; 
        left_transformStamped.child_frame_id = "left";  

        left_transformStamped.transform.translation.x = 0.0;
        left_transformStamped.transform.translation.y = 0; // Offset in y-axis (left-right translation)
        left_transformStamped.transform.translation.z = 0.0;

        left_transformStamped.transform.rotation.x = 0.0;
        left_transformStamped.transform.rotation.y = 0.0;
        left_transformStamped.transform.rotation.z = 0.0;
        left_transformStamped.transform.rotation.w = 1.0;

        tf_broadcaster_->sendTransform(left_transformStamped);

        geometry_msgs::msg::TransformStamped right_transformStamped;

        right_transformStamped.header.stamp = this->now();
        right_transformStamped.header.frame_id = "camera_link"; 
        right_transformStamped.child_frame_id = "right";  

        right_transformStamped.transform.translation.x = 0.0;
        right_transformStamped.transform.translation.y = 0; 
        right_transformStamped.transform.translation.z = 0.0;

        right_transformStamped.transform.rotation.x = 0.0;
        right_transformStamped.transform.rotation.y = 0.0;
        right_transformStamped.transform.rotation.z = 0.0;
        right_transformStamped.transform.rotation.w = 1.0;

        tf_broadcaster_->sendTransform(right_transformStamped);
    }

    std::shared_ptr<ModelRunner> model_runner_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_image_publisher_;  
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_publisher_;  // Camera info publisher
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscriber_;  // Camera info subscriber
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    sensor_msgs::msg::CameraInfo::SharedPtr last_camera_info_;  // Cached camera info
    rclcpp::Time time;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SlamNode>());
    rclcpp::shutdown();
    return 0;
}
