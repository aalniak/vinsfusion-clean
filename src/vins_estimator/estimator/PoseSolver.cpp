/**
 * PoseSolver.cpp
 * * Implements relative pose estimation using SuperPoint+LightGlue (TensorRT)
 * and PoseLib's 3-point monocular depth solver.
 * * Key Features:
 * - Handles resolution mismatch between TensorRT (1280x800) and VINS.
 * - Handles Affine-Invariant Inverse Depth by inverting to "Proxy Depth".
 * - Uses Robust RANSAC estimation.
 */

#include "PoseSolver.h"
#include <iostream>

// PoseLib Headers
#include <PoseLib/PoseLib.h>
#include <PoseLib/robust/robust.h>

// VINS / Eigen Headers
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

// ---------------------------------------------------------------------------
// Helper: Safe Depth Sampling
// ---------------------------------------------------------------------------
// Reads the affine-invariant inverse depth map.
// Returns -1.0f if out of bounds.
float sampleDepthMap(const cv::Mat& depth_map, const cv::Point2f& pt) {
    if (pt.x < 0 || pt.x >= depth_map.cols || pt.y < 0 || pt.y >= depth_map.rows) {
        return -1.0f;
    }
    
    // Check type to handle float vs double maps
    if (depth_map.type() == CV_32F) {
        return depth_map.at<float>(cv::cvRound(pt.y), cv::cvRound(pt.x));
    } else if (depth_map.type() == CV_64F) {
        return static_cast<float>(depth_map.at<double>(cv::cvRound(pt.y), cv::cvRound(pt.x)));
    }
    return -1.0f;
}

// ---------------------------------------------------------------------------
// Main Solver Function
// ---------------------------------------------------------------------------

bool SolvePoseWithMonoDepth(
    const cv::Mat& img1,                // Image 1 (Raw)
    const cv::Mat& img2,                // Image 2 (Raw)
    const cv::Mat& depth_map1,          // Depth Map 1 (Affine-Invariant Inverse)
    SplgInference* matcher,             // Your TensorRT Wrapper
    CameraPtr camera,                   // VINS Camera Model (Abstract Base Class)
    Eigen::Matrix3d& R_est,             // [Output] Rotation
    Eigen::Vector3d& t_est              // [Output] Translation
) {
    // 1. Run SuperPoint + LightGlue Inference
    //    (The matcher handles resizing internally)
    MatchResult matches = matcher->run(img1, img2);

    if (matches.kps1.size() < 10) {
        // std::cout << "[PoseSolver] Not enough matches: " << matches.kps1.size() << std::endl;
        return false;
    }

    // 2. Prepare Data for PoseLib
    //    PoseLib needs: Unit Bearing Vectors and Depth Priors
    std::vector<Eigen::Vector3d> bearings1;
    std::vector<Eigen::Vector3d> bearings2;
    std::vector<double> depth_priors;

    bearings1.reserve(matches.kps1.size());
    bearings2.reserve(matches.kps1.size());
    depth_priors.reserve(matches.kps1.size());

    for (size_t i = 0; i < matches.kps1.size(); ++i) {
        cv::Point2f p1 = matches.kps1[i];
        cv::Point2f p2 = matches.kps2[i];

        // A. Sample Inverse Depth
        float inv_d = sampleDepthMap(depth_map1, p1);

        // B. Validation / Filtering
        //    inv_d <= 1e-5 means infinite depth or void
        if (inv_d <= 1e-5) continue; 

        // C. Convert to Proxy Depth
        //    Since depth is affine-invariant inverse: P = a * (1/Z) + b
        //    We input 1/P to PoseLib, and it solves for scale/shift to fix it.
        double d_proxy = 1.0 / static_cast<double>(inv_d);

        //    Filter extremely large proxy depths (likely sky/noise)
        if (d_proxy > 1000.0) continue;

        // D. Lift Pixels to Bearing Vectors
        Eigen::Vector3d b1, b2;
        //    Note: VINS liftProjective usually handles distortion if calibrated
        camera->liftProjective(Eigen::Vector2d(p1.x, p1.y), b1);
        camera->liftProjective(Eigen::Vector2d(p2.x, p2.y), b2);

        bearings1.push_back(b1.normalized());
        bearings2.push_back(b2.normalized());
        depth_priors.push_back(d_proxy);
    }

    // Need minimal points (3) + buffer for RANSAC
    if (bearings1.size() < 10) {
        // std::cout << "[PoseSolver] Not enough valid depth-bearing pairs." << std::endl;
        return false;
    }

    // 3. Configure PoseLib Robust Estimator
    poselib::CameraPose best_pose;
    poselib::RansacOptions ransac_opt;
    poselib::BundleOptions bundle_opt;
    
    // Tunable Parameters
    // Approx 1.5 pixels error threshold. 
    // If focal length ~460, threshold = 1.5/460 = 0.0032
    ransac_opt.max_reproj_error = 1.5 / 460.0; 
    ransac_opt.min_iterations = 1000;
    ransac_opt.max_iterations = 5000;
    ransac_opt.success_prob = 0.9999;
    
    // 4. Run the Solver
    //    Function: estimate_relpose_monodepth
    //    Solves: x2 ~ R * x1 + t/depth
    //    And:    depth_metric = alpha * depth_prior + beta
    poselib::RansacStats stats = poselib::estimate_relpose_monodepth(
        bearings1, 
        bearings2, 
        depth_priors, 
        ransac_opt, 
        bundle_opt, 
        &best_pose
    );

    // 5. Evaluate Result
    if (stats.num_inliers < 20) {
        // std::cout << "[PoseSolver] RANSAC failed. Inliers: " << stats.num_inliers << std::endl;
        return false;
    }

    // Success! Copy outputs.
    R_est = best_pose.R();
    t_est = best_pose.t();

    // Debugging (Optional)
    /*
    std::cout << "[PoseSolver] Success!" << std::endl;
    std::cout << "  Inliers: " << stats.num_inliers << "/" << bearings1.size() << std::endl;
    std::cout << "  T_norm:  " << t_est.norm() << std::endl;
    */

    return true;
}