#include "models.hpp"
#include <stdexcept>

UniDepthModel::UniDepthModel(const std::string& modelPath, int imageWidth, int imageHeight)
    : BaseModel(modelPath, imageWidth, imageHeight), env_(ORT_LOGGING_LEVEL_WARNING, "UniDepthInference"), session_(nullptr){
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(2);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));
    session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions);
}

cv::Mat UniDepthModel::preprocess(const cv::Mat& inputImage) {
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

cv::Mat UniDepthModel::inference(const cv::Mat& preprocessedImage) {
    const char* inputNames[] = {"image"};
    std::vector<int64_t> inputDims = {1, 3, imageHeight_, imageWidth_};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> inputData((float*)preprocessedImage.datastart, (float*)preprocessedImage.dataend);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputData.size(), inputDims.data(), inputDims.size());

    std::array<Ort::Value*, 1> inputTensors = {&inputTensor};

    const char* outputNames[] = {"depth"};

    auto outputTensors = session_.Run(Ort::RunOptions{nullptr}, inputNames, 
                                        *inputTensors.data(), 
                                        inputTensors.size(), outputNames, 1);

    float* outputData = outputTensors.front().GetTensorMutableData<float>();
    cv::Mat depthMat(imageHeight_, imageWidth_, CV_32FC1, outputData);
    return depthMat.clone(); 
}


// -----------------------------------------------------------------


DepthProModel::DepthProModel(const std::string& modelPath, int imageWidth, int imageHeight)
    : BaseModel(modelPath, imageWidth, imageHeight), env_(ORT_LOGGING_LEVEL_WARNING, "DepthProInference"), session_(nullptr){
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(2);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));
    session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions);
}

cv::Mat DepthProModel::preprocess(const cv::Mat& inputImage) {
    cv::Mat imageRgb;
    cv::cvtColor(inputImage, imageRgb, cv::COLOR_BGR2RGB);
    
    #ifdef SIZE_DEBUG
    std::cout << "Size after color conversion (imageRgb): " << imageRgb.size() << std::endl;
    #endif

    imageRgb.convertTo(imageRgb, CV_32FC3, 1.0/255.0);
    cv::Vec3f mean = cv::Vec3f(0, 0.5, 0.5);
    cv::Vec3f stddev = cv::Vec3f(0.5, 0.5, 0.5);
    
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

cv::Mat DepthProModel::inference(const cv::Mat& preprocessedImage) {
    const char* inputNames[] = {"pixel_values"};
    std::vector<int64_t> inputDims = {1, 3, imageHeight_, imageWidth_};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Convert input data to FP32, assuming FP16 input initially
    std::vector<float> inputData(preprocessedImage.rows * preprocessedImage.cols * 3);
    if (preprocessedImage.type() == CV_16F) {
        cv::Mat preprocessedImageFP32;
        preprocessedImage.convertTo(preprocessedImageFP32, CV_32F);  // Convert FP16 to FP32
        inputData.assign((float*)preprocessedImageFP32.datastart, (float*)preprocessedImageFP32.dataend);
    } else {
        inputData.assign((float*)preprocessedImage.datastart, (float*)preprocessedImage.dataend);
    }

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputData.size(), inputDims.data(), inputDims.size());
    std::array<Ort::Value*, 1> inputTensors = {&inputTensor};

    const char* outputNames[] = {"predicted_depth", "focallength_px"};
    auto outputTensors = session_.Run(Ort::RunOptions{nullptr}, inputNames, 
                                        *inputTensors.data(), 
                                        inputTensors.size(), outputNames, 2);

    // Process inverse depth
    cv::Mat depthMat;
    Ort::Value& depthTensor = outputTensors[0];
    if (depthTensor.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        // Convert FP16 output to FP32
        std::vector<Ort::Float16_t> outputDataFP16(depthTensor.GetTensorMutableData<Ort::Float16_t>(),
                                                   depthTensor.GetTensorMutableData<Ort::Float16_t>() + 
                                                   depthTensor.GetTensorTypeAndShapeInfo().GetElementCount());
        std::vector<float> outputDataFP32(outputDataFP16.size());

        depthMat = cv::Mat(imageHeight_, imageWidth_, CV_32FC1, outputDataFP16.data());
    } else {
        float* outputData = depthTensor.GetTensorMutableData<float>();
        depthMat = cv::Mat(imageHeight_, imageWidth_, CV_32FC1, outputData);
    }

    // Process focal length output
    Ort::Value& focalLengthTensor = outputTensors[1];
    float focalLengthPx = *focalLengthTensor.GetTensorMutableData<float>();

    return depthMat; 
}
