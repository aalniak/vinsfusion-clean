#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "SplgInference.h"
#include "PoseSolver.h" // The file where we put SolvePoseWithMonoDepth
#include "parameters.h" // For CameraPtr

// Structure to hold the input for the thread
struct SolverTask {
    cv::Mat img1;
    cv::Mat img2;
    cv::Mat depth1;
    double timestamp; // To identify which frame this result belongs to
};

// Structure to hold the output
struct SolverResult {
    double timestamp;
    bool success;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
};

class AsyncPoseSolver {
public:
    AsyncPoseSolver();
    ~AsyncPoseSolver();

    // Call this from VINS main thread (processImage)
    // Returns true if task was added, false if dropped (busy)
    bool requestPose(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& depth1, double timestamp);

    // Call this to check if a result is ready
    bool getResult(SolverResult& out_result);

private:
    void workerLoop();

    // Resources
    SplgInference* matcher_;
    CameraPtr camera_; // Make sure this is initialized!

    // Threading primitives
    std::thread worker_thread_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> running_;
    
    // Data buffers
    std::queue<SolverTask> task_queue_;
    std::queue<SolverResult> result_queue_;

    // Performance settings
    const size_t MAX_QUEUE_SIZE = 2; // Drop frames if solver can't keep up
};