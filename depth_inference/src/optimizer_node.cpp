// Robust ROS 2 Node for IMU-RTAB Odometry Fusion using GTSAM (KITTI Dataset Optimized)

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuBias.h>
#include <Eigen/Dense>

#define WINDOW_SIZE 10

using namespace std::placeholders;

class IMUFusionNode : public rclcpp::Node {
public:
    IMUFusionNode() : Node("imu_fusion_node"), initialized_(false), key_index_(0) {
        auto imu_params = gtsam::PreintegrationCombinedParams::MakeSharedU(9.81);
        imu_params->accelerometerCovariance = I_3x3 * pow(0.0001, 2);
        imu_params->gyroscopeCovariance = I_3x3 * pow(0.00001, 2);
        imu_params->integrationCovariance = I_3x3 * pow(1e-4, 2);
        imu_params->biasAccCovariance = I_3x3 * pow(1e-5, 2);
        imu_params->biasOmegaCovariance = I_3x3 * pow(1e-5, 2);

        bias_ = gtsam::imuBias::ConstantBias();
        imu_integrator_ = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(imu_params, bias_);

        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>("/kitti/oxts/imu", 100, bind(&IMUFusionNode::imuCallback, this, _1));
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>("/odom", 10, bind(&IMUFusionNode::odomCallback, this, _1));
        fused_pub_ = create_publisher<nav_msgs::msg::Odometry>("/optimized_odom", 10);

        gtsam::ISAM2Params isam_params;
        optimizer_ = std::make_shared<gtsam::ISAM2>(isam_params);
    }

private:
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        if (!initialized_) return;

        double dt = timestamp - last_imu_time_;
        if (dt <= 0) return;

        imu_integrator_->integrateMeasurement(
            {msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z},
            {msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z},
            dt);

        last_imu_time_ = timestamp;
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        gtsam::Symbol X('x', key_index_), V('v', key_index_), B('b', key_index_);
        gtsam::Pose3 pose(
            gtsam::Rot3::Quaternion(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                                    msg->pose.pose.orientation.y, msg->pose.pose.orientation.z),
            gtsam::Point3(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z));

        gtsam::Vector3 velocity(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);

        if (!initialized_) {
            auto pose_prior_noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(1e-3));
            auto vel_prior_noise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-3);
            auto bias_prior_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);

            graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X, pose, pose_prior_noise));
            graph_.add(gtsam::PriorFactor<gtsam::Vector3>(V, velocity, vel_prior_noise));
            graph_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B, bias_, bias_prior_noise));

            values_.insert(X, pose);
            values_.insert(V, velocity);
            values_.insert(B, bias_);

            optimizer_->update(graph_, values_);

            graph_.resize(0); values_.clear();

            initialized_ = true;
            last_imu_time_ = timestamp;
            key_index_++;
            return;
        }

        if (imu_integrator_->deltaTij() < 0.001) return;

        gtsam::Symbol prev_X('x', key_index_ - 1), prev_V('v', key_index_ - 1), prev_B('b', key_index_ - 1);
        graph_.add(gtsam::CombinedImuFactor(prev_X, prev_V, X, V, prev_B, B, *imu_integrator_));

        auto bias_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
        graph_.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(prev_B, B, {}, bias_noise));

        auto vel_smooth_noise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-1);
        graph_.add(gtsam::BetweenFactor<gtsam::Vector3>(prev_V, V, gtsam::Vector3::Zero(), vel_smooth_noise));

        values_.insert(X, pose);
        values_.insert(V, velocity);
        values_.insert(B, bias_);

        optimizer_->update(graph_, values_);
        if (key_index_ % WINDOW_SIZE == 0) {
            optimizer_->update();
            // optimizer_->marginalizeLeaves();
        }

        auto result = optimizer_->calculateEstimate();

        bias_ = result.at<gtsam::imuBias::ConstantBias>(B);

        nav_msgs::msg::Odometry odom_out;
        odom_out.header = msg->header;
        odom_out.header.frame_id = "odom";
        odom_out.child_frame_id = "base_link";

        auto opt_pose = result.at<gtsam::Pose3>(X);
        auto opt_vel = result.at<gtsam::Vector3>(V);

        odom_out.pose.pose.position.x = opt_pose.x();
        odom_out.pose.pose.position.y = opt_pose.y();
        odom_out.pose.pose.position.z = opt_pose.z();

        auto q = opt_pose.rotation().toQuaternion();
        odom_out.pose.pose.orientation.x = q.x();
        odom_out.pose.pose.orientation.y = q.y();
        odom_out.pose.pose.orientation.z = q.z();
        odom_out.pose.pose.orientation.w = q.w();

        odom_out.twist.twist.linear.x = opt_vel.x();
        odom_out.twist.twist.linear.y = opt_vel.y();
        odom_out.twist.twist.linear.z = opt_vel.z();


        fused_pub_->publish(odom_out);

        graph_.resize(0); values_.clear(); imu_integrator_->resetIntegration();
        key_index_++;
    }

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr fused_pub_;

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    std::shared_ptr<gtsam::ISAM2> optimizer_;
    std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> imu_integrator_;
    gtsam::imuBias::ConstantBias bias_;

    bool initialized_;
    double last_imu_time_;
    size_t key_index_;

    const Eigen::Matrix3d I_3x3 = Eigen::Matrix3d::Identity();
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<IMUFusionNode>());
    rclcpp::shutdown();
    return 0;
}
