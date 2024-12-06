#ifndef UNIDEPTHMODEL_HPP
#define UNIDEPTHMODEL_HPP

#include "base_model.hpp"
#include <onnxruntime/onnxruntime_cxx_api.h>

class UniDepthModel : public BaseModel {
public:
    UniDepthModel(const std::string& modelPath, int imageWidth, int imageHeight);

    cv::Mat preprocess(const cv::Mat& inputImage) override;
    cv::Mat inference(const cv::Mat& preprocessedImage) override;

private:
    Ort::Env env_;
    Ort::Session session_;
};

class DepthProModel : public BaseModel {
public:
    DepthProModel(const std::string& modelPath, int imageWidth, int imageHeight);

    cv::Mat preprocess(const cv::Mat& inputImage) override;
    cv::Mat inference(const cv::Mat& preprocessedImage) override;

private:
    Ort::Env env_;
    Ort::Session session_;
};

class Metric3DModel : public BaseModel {
public:
    Metric3DModel(const std::string& modelPath, int imageWidth, int imageHeight);

    cv::Mat preprocess(const cv::Mat& inputImage) override;
    cv::Mat inference(const cv::Mat& preprocessedImage) override;

private:
    Ort::Env env_;
    Ort::Session session_;
};
#endif // UNIDEPTHMODEL_HPP
