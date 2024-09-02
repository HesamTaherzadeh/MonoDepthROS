#include "model_reader.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>

class SlamNode : public rclcpp::Node {
public:
    SlamNode() : Node("slam_node") {
        model_runner_ = std::make_shared<ModelRunner>("/home/hesam/Desktop/playground/depth_node/src/unidepthv2_vits14_simp.onnx", 644, 364);

        depth_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("depth_image", 10);

        image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera2/left/image_raw", 10,
            std::bind(&SlamNode::image_callback, this, std::placeholders::_1)
        );
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        // Convert ROS image message to OpenCV image using the correct encoding
        cv::Mat input_image = cv_bridge::toCvShare(msg)->image;

        if (input_image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Received an empty image!");
            return;
        }

        // Run inference to get the depth image
        cv::Mat depth_image = model_runner_->runInference(input_image);

        // Create a ROS message header
        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = "camera_link";

        // Convert OpenCV depth image to ROS Image message
        sensor_msgs::msg::Image::SharedPtr depth_msg = cv_bridge::CvImage(
            header, 
            sensor_msgs::image_encodings::TYPE_8UC1, 
            depth_image
        ).toImageMsg();

        // Publish the depth image
        depth_image_publisher_->publish(*depth_msg);
    }

    std::shared_ptr<ModelRunner> model_runner_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SlamNode>());
    rclcpp::shutdown();
    return 0;
}
