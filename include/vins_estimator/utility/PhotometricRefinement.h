#pragma once

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <map>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vins_estimator/factor/photometricRegFactor.h>
#include <vins_estimator/factor/PhotometricSparseFactor.h>
#include <vins_estimator/utility/utility.h>
#include <vector>

namespace vins::estimator {

// Structure to hold optimization results
struct RefinementResult {
    double timestamp_ref;  // i
    double timestamp_cur;  // j
    Eigen::Vector3d t_ref_cur; // Computed relative translation
    Eigen::Quaterniond q_ref_cur; // Computed relative rotation
    Eigen::Matrix<double, 6, 6> information; // Information matrix (Hessian) of the result
    bool success;
    
    // Extra params
    double alpha = 1.0, beta = 0.0;
    double scale = 1.0, shift = 0.0;
    int feature_count = 0; // Debug
};

struct FeaturePoint {
    Eigen::Vector2d u; // pixel coordinate
    double intensity;
    double inv_depth;
};

class PhotometricRefinement {
public:
    enum Mode {
        POSE_ONLY,
        POSE_AFFINE,
        POSE_SCALE_SHIFT,
        POSE_FULL
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PhotometricRefinement();
    ~PhotometricRefinement();

    // Set Refinement Mode
    void setMode(Mode mode) { mode_ = mode; }
    Mode getMode() const { return mode_; }

    // Submit a task to the refinement thread
    void submitTask(double t_ref, double t_cur, 
                    const cv::Mat& img_ref, const cv::Mat& img_cur, 
                    const cv::Mat& depth_ref, const Eigen::Matrix3d& K,
                    const Eigen::Quaterniond& q_initial, const Eigen::Vector3d& t_initial);
    
    // Clear pending tasks (e.g. when starting a new optimization iteration)
    void clearQueues();
    
    // Retrieve results (non-blocking)
    bool getResult(RefinementResult& result);
    
    // Sparse Feature Selection
    // Selects pixels with high gradient, distributed in grid
    void selectPixelFeatures(const cv::Mat& img, const cv::Mat& depth, 
                             std::vector<FeaturePoint>& features,
                             int grid_size = 32, int features_per_grid = 1);
    
    // Depth-Percentile Feature Selection
    // Selects pixels whose fitted (scaled/shifted) inverse depth is in the 40%-70% percentile range
    void selectPixelFeaturesByDepthPercentile(const cv::Mat& img, const cv::Mat& depth,
                                               std::vector<FeaturePoint>& features,
                                               double scale = 1.0, double shift = 0.0,
                                               double lower_percentile = 0.4,
                                               double upper_percentile = 0.7);

private:
    struct Task {
        double t_ref, t_cur;
        cv::Mat img_ref, img_cur;
        cv::Mat depth_ref;
        Eigen::Matrix3d K;
        Eigen::Quaterniond q_initial; // T_cur_ref_initial (from VINS prediction)
        Eigen::Vector3d t_initial;
    };

    // Optimization Parameters
    double ssim_weight_ = 0.85;
    double l1_weight_ = 0.15;
    double huber_loss_ = 15.0; // Huber threshold on raw intensity difference
    
    Mode mode_ = POSE_FULL;
    bool use_sparse_ = true; // Toggle for sparse vs dense
    
    // --- Hand-rolled Gauss-Newton Solver ---
/* struct GNResult {
    // Pose results
    Eigen::Quaterniond q_cur_ref;
    Eigen::Vector3d t_cur_ref;
    
    // Affine results (alpha * I + beta)
    double alpha = 1.0;
    double beta = 0.0;
    
    // Geometric results (scale * d + shift)
    double scale = 1.0;
    double shift = 0.0;

    // Solver stats
    double initial_cost = 0;
    double final_cost = 0;
    int iterations = 0;
    int num_valid = 0;

    // Information matrix (Hessian). 
    // Max size 10x10 (6 pose + 2 affine + 2 scale/shift)
    Eigen::Matrix<double, 10, 10> H_full = Eigen::Matrix<double, 10, 10>::Zero();
}; */
struct GNResult {
        // Pose
        Eigen::Quaterniond q_cur_ref;
        Eigen::Vector3d t_cur_ref;
        
        // Affine
        double alpha = 1.0;
        double beta = 0.0;
        
        // Scale/Shift
        double scale = 1.0;
        double shift = 0.0;

        // Stats
        double initial_cost = 0;
        double final_cost = 0;
        int iterations = 0;
        int num_valid = 0;

        // Information matrix (Max size 10x10: 6 pose + 2 affine + 2 scale/shift)
        Eigen::Matrix<double, 10, 10> H_full = Eigen::Matrix<double, 10, 10>::Zero();
        
        // Helper to get just the 6x6 pose part (for compatibility)
        Eigen::Matrix<double, 6, 6> H_pose() const {
            return H_full.topLeftCorner<6, 6>();
        }
    };
    
    /* GNResult solvePhotometricGN(
        const std::vector<FeaturePoint>& features,
        const cv::Mat& img_cur,
        const Eigen::Matrix3d& K,
        const Eigen::Quaterniond& q_init,
        const Eigen::Vector3d& t_init,
        int max_iterations = 20);
     */
     GNResult solvePhotometricGN_Extended(
        const std::vector<FeaturePoint>& features,
        const cv::Mat& img_cur,
        const Eigen::Matrix3d& K,
        const Eigen::Quaterniond& q_init,
        const Eigen::Vector3d& t_init,
        int max_iterations,
        bool opt_affine,
        bool opt_scaleshift,
        const double* init_affine = nullptr,
        const double* init_scaleshift = nullptr
    );
    // Dual-thread architecture to prevent frame aging
    void threadLoop0();  // Worker thread 0
    void threadLoop1();  // Worker thread 1
    
    std::queue<Task> task_queue_0_;
    std::queue<Task> task_queue_1_;
    std::queue<RefinementResult> result_queue_0_;
    std::queue<RefinementResult> result_queue_1_;
    
    std::mutex task_mutex_0_;
    std::mutex task_mutex_1_;
    std::mutex result_mutex_0_;
    std::mutex result_mutex_1_;
    std::condition_variable task_cond_0_;
    std::condition_variable task_cond_1_;
    
    std::thread processing_thread_0_;
    std::thread processing_thread_1_;
    std::atomic<bool> thread_0_busy_{false};
    std::atomic<bool> thread_1_busy_{false};
    std::atomic<bool> keep_running_;

};

} // namespace vins::estimator
