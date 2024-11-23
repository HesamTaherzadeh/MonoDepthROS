#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/camera_info.hpp>
#include <string>

enum class Dataset
{
    KITTI,
    EUROC,
    NYU,
    TUM,
    MYNTEYE // Added MYNTEYE option for custom camera info
};

class CameraInfoPublisher : public rclcpp::Node
{
public:
    // Constructor takes dataset type as input
    CameraInfoPublisher()
        : Node("mynteye")
    {
        this->declare_parameter("dataset", "KITTI");
        std::string dataset_name = this->get_parameter("dataset").as_string();

        publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_params", 10);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&CameraInfoPublisher::publishCameraInfo, this));

        RCLCPP_INFO(this->get_logger(), "Dataset : %s", dataset_name.c_str());

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

        case Dataset::MYNTEYE: // Handle MYNTEYE dataset
            camera_info_msg.width = 640;
            camera_info_msg.height = 480;
            camera_info_msg.distortion_model = "plumb_bob";
            camera_info_msg.k = {354.12091064453125, 0.0, 321.9713439941406, 0.0, 354.77532958984375, 244.30653381347656, 0.0, 0.0, 1.0};
            camera_info_msg.d = {-0.3059272766113281, 0.08570480346679688, 0.000270843505859375, 0.000888824462890625, 0.0};
            camera_info_msg.r = {0.9999494552612305, -0.003957867622375488, 0.009237408638000488,
                                 0.003968477249145508, 0.9999914169311523, -0.0011354684829711914,
                                 -0.009232878684997559, 0.0011720657348632812, 0.9999566078186035};
            camera_info_msg.p = {349.199951171875, 0.0, 315.102783203125, 
                                 0.0, 0.0, 349.199951171875,
                                 242.9564208984375, 0.0, 0.0, 0.0, 1.0, 0.0};
            camera_info_msg.binning_x = 0;
            camera_info_msg.binning_y = 0;
            camera_info_msg.roi.x_offset = 0;
            camera_info_msg.roi.y_offset = 0;
            camera_info_msg.roi.height = 0;
            camera_info_msg.roi.width = 0;
            camera_info_msg.roi.do_rectify = false;
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
        else if (dataset_name == "MYNTEYE")
            return Dataset::MYNTEYE; // Handle MYNTEYE dataset
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
