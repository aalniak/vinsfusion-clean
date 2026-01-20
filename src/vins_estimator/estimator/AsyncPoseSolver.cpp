#include "vins_estimator/estimator/AsyncPoseSolver.h"

AsyncPoseSolver::AsyncPoseSolver() : running_(true) {
    // Initialize TensorRT Matcher
    // Ensure the path is correct
    matcher_ = new SplgInference("/datasets/splg_1280x800_fp16.engine");
    
    // Initialize Camera (Usually global in VINS, or passed in constructor)
    // m_camera is typically defined in parameters.cpp in VINS
    extern CameraPtr m_camera; 
    camera_ = m_camera;

    // Start the thread
    worker_thread_ = std::thread(&AsyncPoseSolver::workerLoop, this);
}

AsyncPoseSolver::~AsyncPoseSolver() {
    // Graceful shutdown
    running_ = false;
    condition_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    delete matcher_;
}

bool AsyncPoseSolver::requestPose(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& depth1, double timestamp) {
    std::unique_lock<std::mutex> lock(queue_mutex_);

    // DROP STRATEGY: If the solver is backed up, don't pile up old frames.
    // Better to skip this request and process the newest data next time.
    if (task_queue_.size() >= MAX_QUEUE_SIZE) {
        return false; 
    }

    SolverTask task;
    // DEEP COPY IS MANDATORY HERE
    task.img1 = img1.clone(); 
    task.img2 = img2.clone();
    task.depth1 = depth1.clone();
    task.timestamp = timestamp;

    task_queue_.push(task);
    lock.unlock();
    condition_.notify_one(); // Wake up worker
    return true;
}

bool AsyncPoseSolver::getResult(SolverResult& out_result) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (result_queue_.empty()) {
        return false;
    }
    out_result = result_queue_.front();
    result_queue_.pop();
    return true;
}

void AsyncPoseSolver::workerLoop() {
    while (running_) {
        SolverTask task;

        // 1. Wait for a task
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return !task_queue_.empty() || !running_; });

            if (!running_) break;

            task = task_queue_.front();
            task_queue_.pop();
        }

        // 2. Perform Heavy Computation (TensorRT + RANSAC)
        Eigen::Matrix3d R_est;
        Eigen::Vector3d t_est;
        
        // Call the function we wrote earlier in PoseSolver.cpp
        bool success = SolvePoseWithMonoDepth(
            task.img1, 
            task.img2, 
            task.depth1, 
            matcher_, 
            camera_, 
            R_est, 
            t_est
        );

        // 3. Store Result
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            SolverResult res;
            res.timestamp = task.timestamp;
            res.success = success;
            if (success) {
                res.R = R_est;
                res.t = t_est;
            }
            result_queue_.push(res);
        }
    }
}