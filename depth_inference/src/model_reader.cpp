#include "model_reader.hpp"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <map>


const std::map<std::string, ModelConfig> ModelRunner::modelConfigs_ = {
    {"unidepth", {{"image"}, {"depth"}, true}},
    {"depth_pro", {{"pixel_values"}, {"predicted_depth", "focallength_px"}, false}},
};

ModelRunner::ModelRunner(const char* modelPath, int imageWidth, int imageHeight, const std::string& modelType) 
    : env_(ORT_LOGGING_LEVEL_WARNING, "DepthInference"), 
      session_(nullptr),  
      allocator_(),  
      imageWidth_(imageWidth),
      imageHeight_(imageHeight) {

    // Get model configuration
    auto it = modelConfigs_.find(modelType);
    if (it == modelConfigs_.end()) {
        throw std::runtime_error("Unsupported model type: " + modelType);
    }
    modelConfig_ = it->second;

    // Create session options
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(2);

    K = (cv::Mat_<float>(3, 3) << 
    718.856, 0.0, 607.1928,
    0.0, 718.856, 185.2157,
    0.0, 0.0, 1.0);

    // Enable the CUDA execution provider
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));

    // Create the session with the CUDA provider
    session_ = Ort::Session(env_, modelPath, sessionOptions);
}

ModelRunner::~ModelRunner() {}

cv::Mat ModelRunner::preprocessImage(const cv::Mat& inputImage) {
    cv::Mat imageRgb;
    cv::cvtColor(inputImage, imageRgb, cv::COLOR_BGR2RGB);
    
    #ifdef SIZE_DEBUG
    std::cout << "Size after color conversion (imageRgb): " << imageRgb.size() << std::endl;
    #endif

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

    #ifdef SIZE_DEBUG
    std::cout << "Size after normalization (imageRgb): " << imageRgb.size() << std::endl;
    #endif

    cv::Mat resizedImage;
    cv::resize(imageRgb, resizedImage, cv::Size(imageWidth_, imageHeight_));

    #ifdef SIZE_DEBUG
    std::cout << "Size after resizing (resizedImage): " << resizedImage.size() << std::endl;
    #endif

    std::vector<cv::Mat> chwChannels(3);
    cv::split(resizedImage, chwChannels);
    cv::Mat chwImage;
    cv::vconcat(chwChannels, chwImage);
    
    #ifdef SIZE_DEBUG
    std::cout << "Size after splitting and concatenation (chwImage): " << chwImage.size() << std::endl;
    #endif

    if (modelConfig_.inputNames.size() > 0 && modelConfig_.inputNames[0] == "input") {
        cv::Mat float16Image;
        chwImage.convertTo(float16Image, CV_16F);
        return float16Image;
    }

    return chwImage;
}

cv::Mat ModelRunner::postprocessDepth(const cv::Mat& depth, int originalWidth, int originalHeight) {  
    double minVal, maxVal;
    cv::minMaxLoc(depth, &minVal, &maxVal);
    cv::Mat depthNormalized;
    std::cout << "Depth Image Min: " << minVal << ", Max: " << maxVal << std::endl;

    cv::Mat depthResized;
    cv::resize(depth, depthResized, cv::Size(originalWidth, originalHeight));

    #ifdef SIZE_DEBUG
    std::cout << "Size after resizing to original dimensions (depthResized): " << depthResized.size() << std::endl;
    std::cout << "-----------------------------------------------------------------------" << std::endl;
    #endif

    return depthResized;
}

cv::Mat ModelRunner::runInference(const cv::Mat& inputImage) {
    cv::Mat preprocessedImage = preprocessImage(inputImage);

    #ifdef SIZE_DEBUG
    std::cout << "Size after preprocessing (preprocessedImage): " << preprocessedImage.size() << std::endl;
    #endif

    // Prepare the input data for the image
    std::vector<Ort::Float16_t> inputData;
    inputData.assign((Ort::Float16_t*)preprocessedImage.datastart, (Ort::Float16_t*)preprocessedImage.dataend);

    std::vector<int64_t> inputDims = {1, 3, imageHeight_, imageWidth_};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<Ort::Float16_t>(
        memoryInfo, inputData.data(), inputData.size() , inputDims.data(), inputDims.size());

    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(inputTensor));

    if (modelConfig_.useKMatrix) {
        cv::Mat kWithBatch;
        K.reshape(1, 1).copyTo(kWithBatch);

        std::vector<float> kData;
        kData.assign((float*)kWithBatch.datastart, (float*)kWithBatch.dataend);

        std::vector<int64_t> kDims = {1, 3, 3};
        Ort::Value kTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, kData.data(), kData.size(), kDims.data(), kDims.size());

        inputTensors.push_back(std::move(kTensor));
    }

    // Convert input and output names to const char* arrays
    std::vector<const char*> inputNames;
    for (const auto& name : modelConfig_.inputNames) {
        inputNames.push_back(name.c_str());
    }

    std::vector<const char*> outputNames;
    for (const auto& name : modelConfig_.outputNames) {
        outputNames.push_back(name.c_str());
    }

    // Run the inference
    std::vector<Ort::Value> outputTensors = session_.Run(
        Ort::RunOptions{nullptr}, 
        inputNames.data(), 
        inputTensors.data(), 
        inputTensors.size(), 
        outputNames.data(), 
        outputNames.size());

    float* outputData = outputTensors.front().GetTensorMutableData<float>();

    if (!outputData || outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount() == 0) {
            std::cerr << "Error: Output data is empty after inference." << std::endl;
            return cv::Mat();   
        }    cv::Mat depthMat(imageHeight_, imageWidth_, CV_32FC1, outputData);



    #ifdef SIZE_DEBUG
    std::cout << "Size after inference (depthMat): " << depthMat.size() << std::endl;
    #endif

    cv::Mat depthResized = postprocessDepth(depthMat, inputImage.cols, inputImage.rows);

    return depthResized;
}
