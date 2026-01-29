#pragma once
#include <string>
#include <vector>
#include <deque>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <atomic>

class DepthInferVideo {
public:
    DepthInferVideo(std::string engine_path);
    ~DepthInferVideo(); 
    
    // Blocking inference (legacy/debug)
    cv::Mat infer(cv::Mat& img);
    
    // Async Interface
    void input_image(double t, const cv::Mat& img);
    cv::Mat get_depth(double t, int timeout_ms=200); // Waits for depth. Default 200ms.
    void get_completed_depths(std::map<double, cv::Mat>& out_map); // Drains all ready depths

private:
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    void* buffers[2]; // 0: Image Input, 1: Depth Output
    
    float* cpu_output_buffer = nullptr;
    float* h_input_pinned = nullptr;
    cudaStream_t stream;
    
    const int INPUT_W = 518;
    const int INPUT_H = 518;
    const int INPUT_SIZE = 3 * 518 * 518;
    const int OUTPUT_W = 518;
    const int OUTPUT_H = 518;
    const int OUTPUT_SIZE = 1 * 518 * 518; 
    
    // State Management
    static const int NUM_STATES = 8;
    static const int CONTEXT_LEN = 31;
    
    struct StateConfig {
        int tokens;
        int channels;
        size_t size_bytes() const { return tokens * channels * sizeof(float); }
    };
    
    // Config based on TRT_CPP_GUIDE
    const StateConfig STATE_CONFIGS[8] = {
        {1369, 1024}, {1369, 1024}, 
        {361, 1024}, {361, 1024},
        {1369, 256}, {1369, 256},
        {5476, 256}, {5476, 256}
    };

    // History: valid state pointers
    std::deque<float*> history[NUM_STATES];
    
    // Pool: recycled buffers
    std::vector<float*> pool[NUM_STATES];
    
    float* allocate_state(int idx);
    void release_state(int idx, float* ptr);
    
    // Temporary Input Buffers (Strided Context)
    void* state_input_buffers[NUM_STATES]; 
    
    // Indices cache to avoid allocation
    std::vector<int> indices_cache;

    // --- Async Workers ---
    std::thread worker_thread;
    std::atomic<bool> running;
    
    std::mutex input_mutex;
    std::condition_variable input_cv;
    std::deque<std::pair<double, cv::Mat>> input_queue;
    
    std::mutex output_mutex;
    std::condition_variable output_cv;
    std::map<double, cv::Mat> output_map; // timestamp -> depth
    
    void worker_loop();
    cv::Mat infer_internal(cv::Mat& img); // Refactored core logic
};

