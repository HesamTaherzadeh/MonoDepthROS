#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include "base_model.hpp"
#include <opencv2/opencv.hpp>
#include <memory>

enum class ModelType {
    UNIDEPTH,
    DEPTH_PRO,
    METRIC_3D
};

inline const std::map<std::string, ModelType> modelTypeMap = {
    {"unidepth", ModelType::UNIDEPTH},
    {"depth_pro", ModelType::DEPTH_PRO},
    {"metric3d", ModelType::METRIC_3D}
};

class Context {
public:
    Context(ModelType modelType, const char* modelPath, int imageWidth, int imageHeight);
    
    void setModel(ModelType modelType);
    cv::Mat runInference(const cv::Mat& inputImage);

private:
    std::unique_ptr<BaseModel> model_;
    void postProcessDepth(cv::Mat& model_output, int input_width, int input_height, cv::Mat& resized_output);
    const char* modelPath_;
    int imageWidth_;
    int imageHeight_;
};

#endif // CONTEXT_HPP
