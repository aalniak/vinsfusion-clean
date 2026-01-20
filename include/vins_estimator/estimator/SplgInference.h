#pragma once

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <NvInfer.h> // TensorRT

// Result structure for VINS usage
struct MatchResult {
    std::vector<cv::Point2f> kps1; // Keypoints in Image 1
    std::vector<cv::Point2f> kps2; // Keypoints in Image 2
    std::vector<float> scores;     // Match confidence
};

class SplgInference {
public:
    // Initialize with path to .engine file
    SplgInference(const std::string& engine_path);
    ~SplgInference();

    // The main function you will call in VINS
    MatchResult run(const cv::Mat& img1, const cv::Mat& img2);

private:
    // TensorRT internal helpers
    bool loadEngine(const std::string& path);
    void prepareBuffers();



    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    // Resolution settings (Must match your Engine)
    const int input_w_ = 1280;
    const int input_h_ = 800;
    
    // GPU Buffers
    void* buffers_[10]; // pointers to GPU memory
    
    // Binding indices
    int idx_img0_, idx_img1_;
    int idx_kpts0_, idx_kpts1_;
    int idx_matches0_, idx_matches1_;
    int idx_scores_;
    // GPU Memory Pointers
    void* d_img0_ = nullptr;
    void* d_img1_ = nullptr;
    void* d_kpts0_ = nullptr;
    void* d_kpts1_ = nullptr;
    void* d_matches_ = nullptr;
    void* d_scores_ = nullptr;
    // CUDA Stream
    void* stream_;
};