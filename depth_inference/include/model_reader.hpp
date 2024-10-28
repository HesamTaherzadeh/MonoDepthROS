/**
 * @file model_runner.hpp
 * @brief Header file for the ModelRunner class.
 * @author Amirhesam Taherzadegani 
 *
 * This file contains the definition of the ModelRunner class, which is responsible for running inference using an ONNX model and processing images with OpenCV.
 */

#ifndef MODEL_READER_HPP
#define MODEL_READER_HPP

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <definitions.h>


struct ModelConfig {
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    bool useKMatrix;
};

/**
 * @class ModelRunner
 * @brief A class to handle ONNX model inference and image preprocessing/postprocessing.
 *
 * The ModelRunner class is designed to load an ONNX model and perform inference on input images. 
 * It also handles image preprocessing and postprocessing tasks using OpenCV.
 */
class ModelRunner {
public:
    /**
     * @brief Constructor for ModelRunner.
     *
     * Initializes the ONNX Runtime environment, loads the model, and sets the input image dimensions.
     * 
     * @param modelPath The path to the ONNX model file.
     * @param imageWidth The width of the input image expected by the model.
     * @param imageHeight The height of the input image expected by the model.
     */
    ModelRunner(const char* modelPath, int imageWidth, int imageHeight, const std::string& modelType);

    /**
     * @brief Destructor for ModelRunner.
     */
    ~ModelRunner();


    /**
     * @brief Runs inference on the input image.
     *
     * Preprocesses the input image, runs the ONNX model inference, and postprocesses the output depth map.
     * 
     * @param inputImage The input image for inference.
     * @return Postprocessed depth map as a cv::Mat.
     */
    cv::Mat runInference(const cv::Mat& inputImage);

private:
/**
     * @brief Postprocesses the depth output from the model.
     *
     * Normalizes the depth data, resizes it back to the original image dimensions.
     * 
     * @param depth The depth map output from the model.
     * @param originalWidth The width of the original input image.
     * @param originalHeight The height of the original input image.
     * @return Postprocessed depth image as a cv::Mat.
     */
    cv::Mat postprocessDepth(const cv::Mat& depth, int originalWidth, int originalHeight);
    
    /**
     * @brief Preprocesses an input image for model inference.
     *
     * Converts the image to RGB, normalizes it, resizes it to the model's expected input size, and rearranges channels.
     * 
     * @param inputImage The input image in BGR format.
     * @return Preprocessed image in CHW format as a cv::Mat.
     */
    cv::Mat preprocessImage(const cv::Mat& inputImage);

    Ort::Env env_;                      /**< ONNX Runtime environment. */
    Ort::Session session_;              /**< ONNX Runtime session for running the model. */
    Ort::AllocatorWithDefaultOptions allocator_; /**< ONNX Runtime memory allocator. */

    int imageWidth_;  /**< The width of the input image expected by the model. */
    int imageHeight_; /**< The height of the input image expected by the model. */
    cv::Mat K;
    std::string model;
    static const std::map<std::string, ModelConfig> modelConfigs_;
    ModelConfig modelConfig_;

    
};

#endif /* MODEL_READER_HPP */
