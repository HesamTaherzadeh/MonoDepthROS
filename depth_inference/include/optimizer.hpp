#pragma once

#if __cplusplus < 201703L
    #error "This code requires C++17 or later. Please update your compiler or use a compiler that supports C++17."
#endif

#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

// GTSAM headers
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuBias.h>

using gtsam::symbol_shorthand::X;

class PoseOptimizer {
    public:
        PoseOptimizer();
        PoseOptimizer(const std::vector<gtsam::Pose3>& vectorPoses);
        
        void clearGraph();

        // Add a prior factor on Pose3
        void addPriorPose3(const gtsam::Pose3& pose, gtsam::Key key, const gtsam::SharedNoiseModel& noiseModel);
        
        bool hasKey(gtsam::Key key) const;
        // Add a between factor on Pose3 using relative pose between initial estimates
        void addBetweenFactorsPose3(gtsam::Key key1, gtsam::Key key2, const gtsam::SharedNoiseModel& noiseModel);

        // Add a between factor on Pose3 with a given between pose
        void addBetweenFactorPose3(gtsam::Key key1, gtsam::Key key2, const gtsam::Pose3& between, const gtsam::SharedNoiseModel& noiseModel);

        // Add a between factor on imuBias::ConstantBias
        void addBetweenFactorBias(gtsam::Key key1, gtsam::Key key2, const gtsam::imuBias::ConstantBias& between, const gtsam::SharedNoiseModel& noiseModel);

        // Set fixed Pose3
        void setFixedPose3(gtsam::Key key, const gtsam::Pose3& pose, const gtsam::SharedNoiseModel& noiseModel);

        // Add prior factor on Vector3 (e.g., for velocity)
        void addPriorFactorVector3(gtsam::Key key, const gtsam::Vector3& variable, const gtsam::SharedNoiseModel& noiseModel);

        // Add prior factor on imuBias::ConstantBias
        void addPriorFactorBias(gtsam::Key key, const gtsam::imuBias::ConstantBias& variable, const gtsam::SharedNoiseModel& noiseModel);

        // Set initial values
        void setInitialValuePose3(gtsam::Key key, const gtsam::Pose3& value);
        void setInitialValueVector3(gtsam::Key key, const gtsam::Vector3& value);
        void setInitialValueBias(gtsam::Key key, const gtsam::imuBias::ConstantBias& value);

        // Add factor to graph
        void addToGraph(const gtsam::NonlinearFactor::shared_ptr& factor);

        void performOptimizationLM(gtsam::Values& values);
        void performOptimizationISAM(gtsam::Values& values, bool clear);
        void performOptimizationDogLeg(gtsam::Values& values);

        boost::shared_ptr<gtsam::Values> getInitialEstimationObject();

        gtsam::NonlinearFactorGraph getGraphObject();

        boost::shared_ptr<gtsam::noiseModel::Diagonal> createConstantDiagonalNoise(int dimension, double sigma);

        void setIMUPreIntegrationParams(const boost::shared_ptr<gtsam::PreintegrationParams>& imuParams);
        void setIMUPreIntegrationParams(double accSTD, double gyroSTD, double integrationSTD);

        void setConstantIMUBias(const gtsam::imuBias::ConstantBias& imuBiasObj);

        void setConstantIMUBias(const gtsam::Vector3& accBiasVector, const gtsam::Vector3& gyroBiasVector);

        // Getter functions
        boost::shared_ptr<gtsam::PreintegrationParams> getIMUParams() const;

        gtsam::imuBias::ConstantBias getIMUBias() const;

    private:
        gtsam::NonlinearFactorGraph graph;
        boost::shared_ptr<gtsam::Values> initialEstimate;
        std::vector<gtsam::Pose3> vectorPoses;
        boost::shared_ptr<gtsam::PreintegrationParams> imuParams;
        gtsam::imuBias::ConstantBias imuBias;
        gtsam::ISAM2 isam;  // ISAM2 optimizer
};
