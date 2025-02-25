#include <memory>
#include <deque>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <Eigen/Dense>
#include <array>
#include <definitions.h>

#include "optimizer.hpp"


#ifdef ENABLE_DEBUG_OPTIMIZER
    #define OPTIM_NODE_DEBUG(msg) \
        std::cout << "[OptimizerNode] "<< ":" << __LINE__ << " - " << (msg) << std::endl;
#else
    #define OPTIM_NODE_DEBUG(msg)
#endif


class OptimizerNode : public rclcpp::Node {
public:
    OptimizerNode()
    : Node("optimizer_node"),
      last_imu_time_(0),
      last_odom_time_(0),
      pose_optimizer_(std::make_shared<PoseOptimizer>()), 
      odom_counter_(0)
    {
        OPTIM_NODE_DEBUG("Entered OptimizerNode constructor.");
        this->declare_parameter ("acc_std", 0.01);  
        this->declare_parameter("gyro_std", 0.001); 
        this->declare_parameter ("integration_std", 0.0001); 
        this->declare_parameter("optimization_interval_ms", 100);
        this->declare_parameter("imu_topic", "/kitti/oxts/imu"); 

        double acc_std = this->get_parameter("acc_std").as_double();
        double gyro_std = this->get_parameter("gyro_std").as_double();
        double integration_std = this->get_parameter("integration_std").as_double();
        int optim_interval_ms = this->get_parameter("optimization_interval_ms").as_int();
        std::string imu_topic = this->get_parameter("imu_topic").as_string();
        

        RCLCPP_INFO(this->get_logger(), "acc_std : %f", acc_std);
        RCLCPP_INFO(this->get_logger(), "gyro_std : %f", gyro_std);
        RCLCPP_INFO(this->get_logger(), "integration_std : %f", integration_std);
        RCLCPP_INFO(this->get_logger(), "optimization_interval_ms : %d", optim_interval_ms);
        RCLCPP_INFO(this->get_logger(), "imu_topic: %s", imu_topic.c_str());

        pose_optimizer_->setIMUPreIntegrationParams(acc_std, gyro_std, integration_std);

        Eigen::Vector3d zero_vector = Eigen::Vector3d::Zero();
        pose_optimizer_->setConstantIMUBias(zero_vector, zero_vector);

        preintegrated_imu_ = std::make_shared<gtsam::PreintegratedImuMeasurements>(
                pose_optimizer_->getIMUParams(), pose_optimizer_->getIMUBias());

        imu_buffer_ = std::make_shared<std::deque<sensor_msgs::msg::Imu>>();

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&OptimizerNode::odomCallback, this, std::placeholders::_1));
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, 100, std::bind(&OptimizerNode::imuCallback, this, std::placeholders::_1));
        optimized_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/optimized_odom", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(optim_interval_ms),
            std::bind(&OptimizerNode::optimize, this));

        OPTIM_NODE_DEBUG("Exited OptimizerNode constructor.");
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        OPTIM_NODE_DEBUG("Entered odomCallback.");

        // Update the current odometry timestamp
        double current_odom_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        // Integrate IMU data between last odometry time and current time
        integrateImuData(last_odom_time_, current_odom_time);

        std::array<double, 36> twist_covariances = msg->twist.covariance; // 6x6 cov matrix 
        std::array<double, 36> pose_covariances = msg->pose.covariance; // 6x6 cov matrix 

        gtsam::Matrix6 pose_cov = gtsam::Matrix6::Zero();
        for (int i = 0; i < 36; ++i) {
            pose_cov(i / 6, i % 6) = pose_covariances[i];
        }

        gtsam::Vector3 vel_cov = gtsam::Vector3::Zero();
        vel_cov(0) = twist_covariances[0];
        vel_cov(1) = twist_covariances[7];
        vel_cov(2) = twist_covariances[14];

        last_odom_msg_ = *msg;
        last_odom_time_ = current_odom_time;

        // Extract pose from odometry message
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

        gtsam::Vector3 linear_vel(
            msg->twist.twist.linear.x, 
            msg->twist.twist.linear.y, 
            msg->twist.twist.linear.z
        );

        gtsam::Key pose_key = gtsam::Symbol('x', odom_counter_);
        gtsam::Key vel_key = gtsam::Symbol('v', odom_counter_);
        gtsam::Key bias_key = gtsam::Symbol('b', odom_counter_);

        OPTIM_NODE_DEBUG("Initializing pose and state variables.");
        pose_optimizer_->setInitialValuePose3(pose_key, pose);
        pose_optimizer_->setInitialValueVector3(vel_key, linear_vel);
        pose_optimizer_->setInitialValueBias(bias_key, pose_optimizer_->getIMUBias());

        if (odom_counter_ == 0) {
            OPTIM_NODE_DEBUG("Adding prior factors for pose, velocity, and bias.");
            auto pose_noise = gtsam::noiseModel::Diagonal::Precisions(pose_cov.diagonal());
            pose_optimizer_->addPriorPose3(pose, pose_key, pose_noise);

            auto vel_noise = gtsam::noiseModel::Isotropic::Precisions(vel_cov);
            Eigen::Vector3d zero_velocity = Eigen::Vector3d::Zero();
            pose_optimizer_->addPriorFactorVector3(vel_key, zero_velocity, vel_noise);

            auto bias_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-7);
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

            preintegrated_imu_->resetIntegration();
        }

        odom_counter_++;

        OPTIM_NODE_DEBUG("Exited odomCallback.");
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        OPTIM_NODE_DEBUG("Entered imuCallback.");

        imu_buffer_->push_back(*msg);

        OPTIM_NODE_DEBUG("Exited imuCallback.");
    }

    void integrateImuData(double start_time, double end_time) {
        OPTIM_NODE_DEBUG("Integrating IMU data.");

        while (!imu_buffer_->empty()) {
            auto imu_msg = imu_buffer_->front();
            double imu_time = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec * 1e-9;

            if (imu_time >= start_time && imu_time <= end_time) {
                double dt = (last_imu_time_ == 0) ? 0 : imu_time - last_imu_time_;
                last_imu_time_ = imu_time;

                if (dt <= 0) {
                    imu_buffer_->pop_front();
                    continue;
                }

                gtsam::Vector3 acc(
                    imu_msg.linear_acceleration.x,
                    imu_msg.linear_acceleration.y,
                    imu_msg.linear_acceleration.z);
                gtsam::Vector3 gyro(
                    imu_msg.angular_velocity.x,
                    imu_msg.angular_velocity.y,
                    imu_msg.angular_velocity.z);

                preintegrated_imu_->integrateMeasurement(acc, gyro, dt);

                imu_buffer_->pop_front();
            } else if (imu_time > end_time) {
                break;
            } else {
                imu_buffer_->pop_front();
            }
        }

        OPTIM_NODE_DEBUG("Completed IMU data integration.");
    }

    void optimize() {
        OPTIM_NODE_DEBUG("Entered optimize.");

        gtsam::Values result;

        try {
            
            pose_optimizer_->performOptimizationISAM(result, true);
            OPTIM_NODE_DEBUG("Performed optimization.");
        } catch (const gtsam::IndeterminantLinearSystemException& e) {
            RCLCPP_ERROR(this->get_logger(), "GTSAM optimization failed: %s", e.what());
            pose_optimizer_->clearGraph();
            return;
        }
        OPTIM_NODE_DEBUG("Performed optimization.");

        if (odom_counter_ > 0) {
            for (size_t i = 0; i < odom_counter_; ++i) {
            gtsam::Key pose_key = gtsam::Symbol('x', i);
            gtsam::Key vel_key = gtsam::Symbol('v', i);

            if (result.exists(pose_key)) {
                gtsam::Pose3 optimized_pose = result.at<gtsam::Pose3>(pose_key);

                auto odom_msg = nav_msgs::msg::Odometry();
                odom_msg.header.stamp = last_odom_msg_.header.stamp; // Adjust timestamp for older states if needed
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

                if (result.exists(vel_key)) {
                    gtsam::Vector3 velocity = result.at<gtsam::Vector3>(vel_key);
                    odom_msg.twist.twist.linear.x = velocity.x();
                    odom_msg.twist.twist.linear.y = velocity.y();
                    odom_msg.twist.twist.linear.z = velocity.z();
                }

                optimized_odom_pub_->publish(odom_msg);
                OPTIM_NODE_DEBUG("Published optimized odometry for key: " + std::to_string(i));
                }
            }
        } else {
            OPTIM_NODE_DEBUG("Odom counter is zero, nothing to optimize.");
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
    double last_odom_time_;
    nav_msgs::msg::Odometry last_odom_msg_;

    std::shared_ptr<gtsam::PreintegratedImuMeasurements> preintegrated_imu_;

    std::shared_ptr<std::deque<sensor_msgs::msg::Imu>> imu_buffer_;
    int count_to_clean;
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
