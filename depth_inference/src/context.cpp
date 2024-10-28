#include "context.hpp"
#include "models.hpp"
#include <stdexcept>
#include <iostream>

Context::Context(ModelType modelType, const char* modelPath, int imageWidth, int imageHeight)
    : modelPath_(modelPath), imageWidth_(imageWidth), imageHeight_(imageHeight) {
    setModel(modelType);  
}

void Context::setModel(ModelType modelType) {
    switch (modelType) {
        case ModelType::UNIDEPTH:
            model_ = std::make_unique<UniDepthModel>(modelPath_, imageWidth_, imageHeight_);
            break;
        case ModelType::DEPTH_PRO:
            model_ = std::make_unique<DepthProModel>(modelPath_, imageWidth_, imageHeight_);
            break;
        default:
            throw std::runtime_error("Unsupported model type.");
    }
}

void Context::postProcessDepth(cv::Mat& model_output, int input_width, int input_height, cv::Mat& resized_output){
    double minVal, maxVal;
    cv::minMaxLoc(model_output, &minVal, &maxVal);
    std::cout << "Depth Image Min: " << minVal << ", Max: " << maxVal << std::endl;

    cv::Mat depthResized;
    cv::resize(model_output, resized_output, cv::Size(input_width, input_height));

    #ifdef SIZE_DEBUG
    std::cout << "Size after resizing to original dimensions (depthResized): " << depthResized.size() << std::endl;
    std::cout << "-----------------------------------------------------------------------" << std::endl;
    #endif
}

cv::Mat Context::runInference(const cv::Mat& inputImage) {
    if (!model_) {
        throw std::runtime_error("Model has not been set.");
    }

    cv::Mat preprocessedImage = model_->preprocess(inputImage);

    #ifdef SIZE_DEBUG
    std::cout << "Size after preprocessing (preprocessedImage): " << preprocessedImage.size() << std::endl;
    #endif

    cv::Mat depth = model_->inference(preprocessedImage);

    if (depth.empty()) {
        std::cerr << "Error: Output data is empty after inference." << std::endl;
        return cv::Mat();  // Return an empty Mat if no data is available
    }

    cv::Mat depthResized ;
    postProcessDepth(depth, inputImage.cols, inputImage.rows, depthResized);
    return depthResized;
}
