#ifndef BASEMODEL_HPP
#define BASEMODEL_HPP

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>

class BaseModel {
public:
    BaseModel(const std::string& modelPath, int imageWidth, int imageHeight)
        : modelPath_(modelPath), imageWidth_(imageWidth), imageHeight_(imageHeight) {}
    virtual ~BaseModel() = default;

    virtual cv::Mat preprocess(const cv::Mat& inputImage) = 0;
    virtual cv::Mat inference(const cv::Mat& preprocessedImage) = 0;

protected:
    std::string modelPath_;
    int imageWidth_;
    int imageHeight_;
};

#endif // BASEMODEL_HPP
