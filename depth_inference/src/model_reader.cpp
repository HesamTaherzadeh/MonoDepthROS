#include "model_reader.hpp"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#define SIZE_DEBUG // Define the macro here or in your build system (e.g., in CMake)

ModelRunner::ModelRunner(const char* modelPath, int imageWidth, int imageHeight) 
    : env_(ORT_LOGGING_LEVEL_WARNING, "DepthInference"), 
      session_(nullptr),  
      allocator_(), 
      imageHeight_(imageHeight), 
      imageWidth_(imageWidth) {

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

    return chwImage;
}

cv::Mat ModelRunner::postprocessDepth(const cv::Mat& depth, int originalWidth, int originalHeight) {  
    double minVal, maxVal;
    cv::minMaxLoc(depth, &minVal, &maxVal);
    cv::Mat depthNormalized;
    depth.convertTo(depthNormalized, CV_8UC1, 255.0 / (maxVal - minVal));

    cv::Mat depthResized;
    cv::resize(depthNormalized, depthResized, cv::Size(originalWidth, originalHeight));

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
    std::vector<float> inputData;
    inputData.assign((float*)preprocessedImage.datastart, (float*)preprocessedImage.dataend);

    std::vector<int64_t> inputDims = {1, 3, imageHeight_, imageWidth_};  // Assuming batch size 1
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputData.size(), inputDims.data(), inputDims.size());

    const char* outputNames[] = {"depth"};

    #ifdef INPUT_K
        cv::Mat kWithBatch;
        K.reshape(1, 1).copyTo(kWithBatch);  

        std::vector<float> kData;
        kData.assign((float*)kWithBatch.datastart, (float*)kWithBatch.dataend);

        std::vector<int64_t> kDims = {1, 3, 3};
        Ort::Value kTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, kData.data(), kData.size(), kDims.data(), kDims.size());

        const char* inputNames[] = {"image", "K"};
        std::array<Ort::Value*, 2> inputTensors = {&inputTensor, &kTensor};

        auto outputTensors = session_.Run(Ort::RunOptions{nullptr}, inputNames, 
                                        *inputTensors.data(), 
                                        2, outputNames, 1);
    #else
        const char* inputNames[] = {"image"};
        std::array<Ort::Value*, 1> inputTensors = {&inputTensor};

        auto outputTensors = session_.Run(Ort::RunOptions{nullptr}, inputNames, 
                                        *inputTensors.data(), 
                                        inputTensors.size(), outputNames, 1);
    #endif

    float* outputData = outputTensors.front().GetTensorMutableData<float>();
    cv::Mat depthMat(imageHeight_, imageWidth_, CV_32FC1, outputData);

    #ifdef SIZE_DEBUG
    std::cout << "Size after inference (depthMat): " << depthMat.size() << std::endl;
    #endif

    cv::Mat depthResized = postprocessDepth(depthMat, inputImage.cols, inputImage.rows);

    return depthResized;
}

