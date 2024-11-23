#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/inference/Symbol.h>
#include <deque>

#include "optimizer.hpp"  // Ensure this includes the updated PoseOptimizer class

#define ENABLE_DEBUG

#ifdef ENABLE_DEBUG
    #define OPTIM_NODE_DEBUG(msg) \
        std::cout << "[Optimizer_node] "<< ":" << __LINE__ << " - " << (msg) << std::endl;
#else
    #define OPTIM_NODE_DEBUG(msg)
#endif

class OptimizerNode : public rclcpp::Node {
public:
    OptimizerNode()
    : Node("optimizer_node"),
      last_imu_time_(0),
      pose_optimizer_(std::make_shared<PoseOptimizer>()), 
      odom_counter_(1)
    {
        OPTIM_NODE_DEBUG("Entered OptimizerNode constructor.");

        preintegrated_imu_ = std::make_shared<gtsam::PreintegratedImuMeasurements>(
                pose_optimizer_->getIMUParams(), pose_optimizer_->getIMUBias());

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&OptimizerNode::odomCallback, this, std::placeholders::_1));
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/kitti/oxts/imu", 100, std::bind(&OptimizerNode::imuCallback, this, std::placeholders::_1));
        double acc_std = 0.01;
        double gyro_std = 0.001;
        double integration_std = 0.0001;
        pose_optimizer_->setIMUPreIntegrationParams(acc_std, gyro_std, integration_std);

        Eigen::Vector3d zero_vector = Eigen::Vector3d::Zero();
        pose_optimizer_->setConstantIMUBias(zero_vector, zero_vector);

        optimized_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/optimized_odom", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10000),
            std::bind(&OptimizerNode::optimize, this));

        OPTIM_NODE_DEBUG("Exited OptimizerNode constructor.");
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        OPTIM_NODE_DEBUG("Entered odomCallback.");

        gtsam::Pose3 pose(
            gtsam::Rot3::Quaternion(
                msg->pose.pose.orientation.w,
                msg->pose.pose.orientation.x,
                msg->pose.pose.orientation.y,
                msg->pose.pose.orientation.z),
            gtsam::Point3(
                msg->pose.pose.position.x,
                msg->pose.pose.position.y,
                msg->pose.pose.position.z));

        gtsam::Key pose_key = gtsam::Symbol('x', odom_counter_);
        gtsam::Key vel_key = gtsam::Symbol('v', odom_counter_);
        gtsam::Key bias_key = gtsam::Symbol('b', odom_counter_);


        std::cout << odom_counter_ << std::endl;

        OPTIM_NODE_DEBUG("INIT pose");
        pose_optimizer_->setInitialValuePose3(pose_key, pose);
        pose_optimizer_->setInitialValueVector3(vel_key, Eigen::Vector3d::Zero());
        pose_optimizer_->setInitialValueBias(bias_key, pose_optimizer_->getIMUBias());

        if (odom_counter_ == 0) {
            OPTIM_NODE_DEBUG("Adding prior factors for pose, velocity, and bias.");
            auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());
            pose_optimizer_->addPriorPose3(pose, pose_key, pose_noise);

            auto vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-3);
            Eigen::Vector3d zero_velocity = Eigen::Vector3d::Zero();
            pose_optimizer_->addPriorFactorVector3(vel_key, zero_velocity, vel_noise);

            auto bias_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
            pose_optimizer_->addPriorFactorBias(bias_key, pose_optimizer_->getIMUBias(), bias_noise);

        } else {
            OPTIM_NODE_DEBUG("Adding IMU and bias factors to the graph.");
            gtsam::Key prev_pose_key = gtsam::Symbol('x', odom_counter_ - 1);
            gtsam::Key prev_vel_key = gtsam::Symbol('v', odom_counter_ - 1);
            gtsam::Key prev_bias_key = gtsam::Symbol('b', odom_counter_ - 1);

            auto imu_factor = boost::make_shared<gtsam::ImuFactor>(
                prev_pose_key, prev_vel_key,
                pose_key, vel_key,
                prev_bias_key, *preintegrated_imu_);

            pose_optimizer_->addToGraph(imu_factor);

            auto bias_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
            auto bias_factor = boost::make_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
                prev_bias_key, bias_key, gtsam::imuBias::ConstantBias(), bias_noise);
            pose_optimizer_->addToGraph(bias_factor);

            preintegrated_imu_->resetIntegrationAndSetBias(pose_optimizer_->getIMUBias());
        }

        odom_counter_++;

        OPTIM_NODE_DEBUG("Exited odomCallback.");
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        OPTIM_NODE_DEBUG("Entered imuCallback.");

        double current_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        if (last_imu_time_ == 0) {
            last_imu_time_ = current_time;
            OPTIM_NODE_DEBUG("First IMU message received. Setting last_imu_time_.");
            return;
        }

        double dt = current_time - last_imu_time_;
        last_imu_time_ = current_time;

        gtsam::Vector3 acc(
            msg->linear_acceleration.x,
            msg->linear_acceleration.y,
            msg->linear_acceleration.z);
        gtsam::Vector3 gyro(
            msg->angular_velocity.x,
            msg->angular_velocity.y,
            msg->angular_velocity.z);

        preintegrated_imu_->integrateMeasurement(acc, gyro, dt);

        OPTIM_NODE_DEBUG("Exited imuCallback.");
    }

    void optimize() {
        OPTIM_NODE_DEBUG("Entered optimize.");

        gtsam::Values result;
        pose_optimizer_->performOptimizationISAM(result);

        if (odom_counter_ > 0) {
            gtsam::Key pose_key = gtsam::Symbol('x', odom_counter_ - 1);
            if (result.exists(pose_key)) {
                gtsam::Pose3 optimized_pose = result.at<gtsam::Pose3>(pose_key);

                auto odom_msg = nav_msgs::msg::Odometry();
                odom_msg.header.stamp = this->get_clock()->now();
                odom_msg.header.frame_id = "odom";
                odom_msg.child_frame_id = "base_link";

                odom_msg.pose.pose.position.x = optimized_pose.x();
                odom_msg.pose.pose.position.y = optimized_pose.y();
                odom_msg.pose.pose.position.z = optimized_pose.z();

                auto quaternion = optimized_pose.rotation().toQuaternion();
                odom_msg.pose.pose.orientation.x = quaternion.x();
                odom_msg.pose.pose.orientation.y = quaternion.y();
                odom_msg.pose.pose.orientation.z = quaternion.z();
                odom_msg.pose.pose.orientation.w = quaternion.w();

                if (result.exists(gtsam::Symbol('v', odom_counter_ - 1))) {
                    gtsam::Vector3 velocity = result.at<gtsam::Vector3>(gtsam::Symbol('v', odom_counter_ - 1));
                    odom_msg.twist.twist.linear.x = velocity.x();
                    odom_msg.twist.twist.linear.y = velocity.y();
                    odom_msg.twist.twist.linear.z = velocity.z();
                }

                optimized_odom_pub_->publish(odom_msg);
            }
        }

        OPTIM_NODE_DEBUG("Exited optimize.");
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr optimized_odom_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::shared_ptr<PoseOptimizer> pose_optimizer_;

    size_t odom_counter_;
    double last_imu_time_;

    std::shared_ptr<gtsam::PreintegratedImuMeasurements> preintegrated_imu_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    OPTIM_NODE_DEBUG("Started OptimizerNode.");
    auto node = std::make_shared<OptimizerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    OPTIM_NODE_DEBUG("Shutdown OptimizerNode.");
    return 0;
}
