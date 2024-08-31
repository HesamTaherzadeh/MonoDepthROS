#include "depth_infer.hpp"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

ModelRunner::ModelRunner(const char* modelPath, int imageWidth, int imageHeight) 
    : env_(ORT_LOGGING_LEVEL_WARNING, "DepthInference"), 
      session_(env_, modelPath, Ort::SessionOptions()), 
      allocator_(), 
      imageHeight_(imageHeight), 
      imageWidth_(imageWidth) {
}

ModelRunner::~ModelRunner() {}

cv::Mat ModelRunner::preprocessImage(const cv::Mat& inputImage) {
    cv::Mat imageRgb;
    cv::cvtColor(inputImage, imageRgb, cv::COLOR_BGR2RGB);

    imageRgb.convertTo(imageRgb, CV_32FC3, 1.0 / 255.0);
    cv::Vec3f mean = cv::Vec3f(0.485, 0.456, 0.406);
    cv::Vec3f stddev = cv::Vec3f(0.229, 0.224, 0.225);
    
    for (int i = 0; i < imageRgb.rows; ++i) {
        for (int j = 0; j < imageRgb.cols; ++j) {
            cv::Vec3f& pixel = imageRgb.at<cv::Vec3f>(i, j);
            pixel[0] = (pixel[0] - mean[0]) / stddev[0];
            pixel[1] = (pixel[1] - mean[1]) / stddev[1];
            pixel[2] = (pixel[2] - mean[2]) / stddev[2];
        }
    }

    cv::Mat resizedImage;
    cv::resize(imageRgb, resizedImage, cv::Size(imageWidth_, imageHeight_));

    std::vector<cv::Mat> chwChannels(3);
    cv::split(resizedImage, chwChannels);
    cv::Mat chwImage;
    cv::vconcat(chwChannels, chwImage);
    
    return chwImage;
}

cv::Mat ModelRunner::postprocessDepth(const cv::Mat& depth, int originalWidth, int originalHeight) {  
    double minVal, maxVal;
    cv::minMaxLoc(depth, &minVal, &maxVal);
    cv::Mat depthNormalized;
    depth.convertTo(depthNormalized, CV_8UC1, 255.0 / (maxVal - minVal));

    cv::Mat depthResized;
    cv::resize(depthNormalized, depthResized, cv::Size(originalWidth, originalHeight));

    return depthResized;
}

cv::Mat ModelRunner::runInference(const cv::Mat& inputImage) {
    cv::Mat preprocessedImage = preprocessImage(inputImage);

    std::vector<float> inputData;
    inputData.assign((float*)preprocessedImage.datastart, (float*)preprocessedImage.dataend);

    std::vector<int64_t> inputDims = {1, 3, imageHeight_, imageWidth_};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputData.size(), inputDims.data(), inputDims.size());

    const char* inputNames[] = {"image"};
    const char* outputNames[] = {"depth"};

    auto outputTensors = session_.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

    float* outputData = outputTensors.front().GetTensorMutableData<float>();
    cv::Mat depthMat(imageHeight_, imageWidth_, CV_32FC1, outputData);

    cv::Mat depthResized = postprocessDepth(depthMat, inputImage.cols, inputImage.rows);

    return depthResized;
}


