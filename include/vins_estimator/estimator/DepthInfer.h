#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class DepthInfer {
public:
    DepthInfer(std::string engine_path);
    ~DepthInfer(); // Destructor declaration
    cv::Mat infer(cv::Mat& img);

private:
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    void* buffers[2]; // 0: Input, 1: Output
    float* cpu_output_buffer = nullptr;
    
    // Hardcoded for Depth Anything AC Small
    const int INPUT_W = 518;
    const int INPUT_H = 518;
    const int INPUT_SIZE = 3 * 518 * 518;
    const int OUTPUT_W = 296;
    const int OUTPUT_H = 296;
    const int OUTPUT_SIZE = 1 * 296 * 296;
};
