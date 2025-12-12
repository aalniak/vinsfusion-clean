#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <opencv2/opencv.hpp>

// Simple Logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} static gLogger;

class NetVLADLoop {
public:
    NetVLADLoop(const std::string& engine_path, int max_db_size = 50000);
    ~NetVLADLoop();

    // 1. Extract: Runs TRT, returns CPU vector (for saving) AND keeps copy on GPU DB
    std::vector<float> extract_and_add(const cv::Mat& img, int frame_index);

    // 2. Query: GPU-accelerated Matrix-Vector Multiplication
    std::pair<int, float> query(int min_interval = 50);

private:
    // TensorRT
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    // cuBLAS (For Math)
    cublasHandle_t cublas_handle;

    // Memory Buffers
    void* buffers[2];          // 0: Input Image, 1: Output Descriptor (TRT)
    float* d_database;         // Giant GPU Buffer for all descriptors
    float* d_scores;           // GPU Buffer for search results
    float* h_scores;           // CPU Buffer to read back results

    // Config
    const int INPUT_W = 320;
    const int INPUT_H = 200;
    const int DESC_DIM = 4096; // 512 * 64
    const int MAX_DB_SIZE = 50000;
    
    // State
    int current_db_size = 0;   // How many frames we have stored
    std::vector<int> db_indices; // Map 0..N to FrameID
};