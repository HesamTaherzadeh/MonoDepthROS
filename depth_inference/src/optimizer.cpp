#include "optimizer.hpp"

PoseOptimizer::PoseOptimizer(): graph(),                                    
      initialEstimate(boost::make_shared<gtsam::Values>()),  
      vectorPoses(),                             
      imuParams(gtsam::PreintegrationParams::MakeSharedD()) {
        imuBias = gtsam::imuBias::ConstantBias();
        gtsam::ISAM2Params params;
        params.relinearizeThreshold = 0.01;
        params.relinearizeSkip = 1;
        isam = gtsam::ISAM2(params);
    }
void PoseOptimizer::addPriorPose3(const gtsam::Pose3& pose, gtsam::Key key, const gtsam::SharedNoiseModel& noiseModel) {
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(key, pose, noiseModel);
}
bool PoseOptimizer::hasKey(gtsam::Key key) const {
    return initialEstimate->exists(key);
}
void PoseOptimizer::addBetweenFactorsPose3(gtsam::Key key1, gtsam::Key key2, const gtsam::SharedNoiseModel& noiseModel) {
    const gtsam::Pose3& pose1 = initialEstimate->at<gtsam::Pose3>(key1);
    const gtsam::Pose3& pose2 = initialEstimate->at<gtsam::Pose3>(key2);
    gtsam::Pose3 relativePose = pose1.between(pose2);
    graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(key1, key2, relativePose, noiseModel);
}

void PoseOptimizer::addBetweenFactorPose3(gtsam::Key key1, gtsam::Key key2, const gtsam::Pose3& between, const gtsam::SharedNoiseModel& noiseModel) {
    graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(key1, key2, between, noiseModel);
}

void PoseOptimizer::addBetweenFactorBias(gtsam::Key key1, gtsam::Key key2, const gtsam::imuBias::ConstantBias& between, const gtsam::SharedNoiseModel& noiseModel) {
    graph.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(key1, key2, between, noiseModel);
}

void PoseOptimizer::setFixedPose3(gtsam::Key key, const gtsam::Pose3& pose, const gtsam::SharedNoiseModel& noiseModel) {
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(key, pose, noiseModel);
}

void PoseOptimizer::addPriorFactorVector3(gtsam::Key key, const gtsam::Vector3& variable, const gtsam::SharedNoiseModel& noiseModel) {
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(key, variable, noiseModel);
}

void PoseOptimizer::addPriorFactorBias(gtsam::Key key, const gtsam::imuBias::ConstantBias& variable, const gtsam::SharedNoiseModel& noiseModel) {
    graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(key, variable, noiseModel);
}

void PoseOptimizer::setInitialValuePose3(gtsam::Key key, const gtsam::Pose3& value) {
    initialEstimate->insert<gtsam::Pose3>(key, value);
}

void PoseOptimizer::setInitialValueVector3(gtsam::Key key, const gtsam::Vector3& value) {
    initialEstimate->insert<gtsam::Vector3>(key, value);
}

void PoseOptimizer::setInitialValueBias(gtsam::Key key, const gtsam::imuBias::ConstantBias& value) {
    initialEstimate->insert<gtsam::imuBias::ConstantBias>(key, value);
}

void PoseOptimizer::addToGraph(const gtsam::NonlinearFactor::shared_ptr& factor) {
    graph.add(factor);
}

void PoseOptimizer::performOptimizationLM(gtsam::Values& values) {
    // Implement Levenberg-Marquardt optimization
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, *initialEstimate);
    values = optimizer.optimize();
}

void PoseOptimizer::performOptimizationISAM(gtsam::Values& values, bool clear) {
    isam.update(graph, *initialEstimate);
    values = isam.calculateEstimate();
    if (clear) clearGraph();         

    std::cout << "Performed iSAM2 optimization and cleared the graph and initial values." << std::endl;
}

void PoseOptimizer::clearGraph(){
    (*initialEstimate).clear();
    graph.resize(0); 
}

void PoseOptimizer::performOptimizationDogLeg(gtsam::Values& values) {
    gtsam::DoglegOptimizer optimizer(graph, *initialEstimate);
    values = optimizer.optimize();
}

boost::shared_ptr<gtsam::Values> PoseOptimizer::getInitialEstimationObject() {
    return initialEstimate;
}

gtsam::NonlinearFactorGraph PoseOptimizer::getGraphObject() {
    return graph;
}

boost::shared_ptr<gtsam::noiseModel::Diagonal> PoseOptimizer::createConstantDiagonalNoise(int dimension, double sigma) {
    gtsam::Vector sigmas = gtsam::Vector::Constant(dimension, sigma);
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

void PoseOptimizer::setIMUPreIntegrationParams(const boost::shared_ptr<gtsam::PreintegrationParams>& imuParams) {
    this->imuParams = imuParams;
}

void PoseOptimizer::setIMUPreIntegrationParams(double accSTD, double gyroSTD, double integrationSTD) {
    imuParams = gtsam::PreintegrationParams::MakeSharedU();
    imuParams->accelerometerCovariance = gtsam::Matrix33::Identity() * accSTD * accSTD;
    imuParams->gyroscopeCovariance = gtsam::Matrix33::Identity() * gyroSTD * gyroSTD;
    imuParams->integrationCovariance = gtsam::Matrix33::Identity() * integrationSTD * integrationSTD;
}

void PoseOptimizer::setConstantIMUBias(const gtsam::imuBias::ConstantBias& imuBiasObj) {
    imuBias = imuBiasObj;
}



void PoseOptimizer::setConstantIMUBias(const gtsam::Vector3& accBiasVector, const gtsam::Vector3& gyroBiasVector){
    imuBias = gtsam::imuBias::ConstantBias(accBiasVector, gyroBiasVector);
}

boost::shared_ptr<gtsam::PreintegrationParams> PoseOptimizer::getIMUParams() const {
    return imuParams;
}

gtsam::imuBias::ConstantBias PoseOptimizer::getIMUBias() const {
    return imuBias;
}
