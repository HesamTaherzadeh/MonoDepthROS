#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/camera_info.hpp>
#include <string>

enum class Dataset
{
    KITTI,
    EUROC,
    NYU,
    TUM
};

class CameraInfoPublisher : public rclcpp::Node
{
public:
    // Constructor takes dataset type as input
    CameraInfoPublisher()
        : Node("utils")
    {

        this->declare_parameter("dataset", "KITTI");
        std::string dataset_name = this->get_parameter("dataset").as_string();

        publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_params", 10);
        timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&CameraInfoPublisher::publishCameraInfo, this));

        dataset_ = parseDatasetName(dataset_name);
    }

private:
    void publishCameraInfo()
    {
        auto camera_info_msg = sensor_msgs::msg::CameraInfo();

        // Set values based on dataset
        setCameraInfo(camera_info_msg, dataset_);

        // Publish the camera info message
        publisher_->publish(camera_info_msg);
    }

    void setCameraInfo(sensor_msgs::msg::CameraInfo &camera_info_msg, Dataset dataset)
    {
        switch (dataset)
        {
        case Dataset::KITTI:
            camera_info_msg.width = 1242;
            camera_info_msg.height = 375;
            camera_info_msg.distortion_model = "plumb_bob";
            camera_info_msg.k = {7.188560000000e+02, 0.0, 6.071928000000e+02, 0.0, 7.188560000000e+02, 1.852157000000e+02, 0.0, 0.0, 1.0};
            break;

        case Dataset::EUROC:
            camera_info_msg.width = 752;
            camera_info_msg.height = 480;
            camera_info_msg.distortion_model = "plumb_bob";
            camera_info_msg.k = {458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0};
            break;

        case Dataset::NYU:
            camera_info_msg.width = 640;
            camera_info_msg.height = 480;
            camera_info_msg.distortion_model = "plumb_bob";
            camera_info_msg.k = {525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0};
            break;

        case Dataset::TUM:
            camera_info_msg.width = 640;
            camera_info_msg.height = 480;
            camera_info_msg.distortion_model = "plumb_bob";
            camera_info_msg.k = {517.3, 0.0, 318.6, 0.0, 516.5, 255.3, 0.0, 0.0, 1.0};
            break;

        default:
            RCLCPP_WARN(this->get_logger(), "Unknown dataset, setting default values.");
            camera_info_msg.width = 640;
            camera_info_msg.height = 480;
            camera_info_msg.distortion_model = "plumb_bob";
            camera_info_msg.k = {525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0};
            break;
        }
    }


    Dataset parseDatasetName(const std::string &dataset_name)
    {
        if (dataset_name == "KITTI")
            return Dataset::KITTI;
        else if (dataset_name == "EUROC")
            return Dataset::EUROC;
        else if (dataset_name == "NYU")
            return Dataset::NYU;
        else if (dataset_name == "TUM")
            return Dataset::TUM;
        else
        {
            RCLCPP_WARN(this->get_logger(), "Invalid dataset name, defaulting to KITTI.");
            return Dataset::KITTI; // Default to KITTI if the input is invalid
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    Dataset dataset_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<CameraInfoPublisher>();

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
