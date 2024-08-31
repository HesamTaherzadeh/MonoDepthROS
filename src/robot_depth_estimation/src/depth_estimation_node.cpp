#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <depth_infer.hpp>

class DepthMapperNode {
private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;
    ros::Publisher depth_map_pub_;

    std::shared_ptr<ModelRunner> model_;
    int width_;
    int height_;

public:
    DepthMapperNode(const std::string& model_path, int width, int height)
        : nh_("~"), width_(width), height_(height) {
        model_ = std::make_shared<ModelRunner>(model_path.c_str(), width_, height_);

        image_sub_ = nh_.subscribe("/cam02/image_raw", 1, &DepthMapperNode::imageCallback, this);
        depth_map_pub_ = nh_.advertise<sensor_msgs::Image>("/depth_map", 1);
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat input_image = cv_ptr->image;

        cv::Mat out = model_->runInference(input_image);

        cv::Mat color_depth_map;
        cv::applyColorMap(out, color_depth_map, cv::COLORMAP_JET);

        sensor_msgs::ImagePtr ros_depth_image;
        try {
            ros_depth_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_depth_map).toImageMsg();
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        depth_map_pub_.publish(ros_depth_image);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "depth_mapping_node");

    const std::string model_path = "/home/user/depth_infer/src/robot_depth_estimation/model/unidepthv2_vits14_simp.onnx";
    int width = 644 ;
    int height = 364;

    DepthMapperNode node(model_path, width, height);

    ros::spin();

    return 0;
}
