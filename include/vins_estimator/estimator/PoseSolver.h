#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "SplgInference.h" // Your TensorRT wrapper
#include "parameters.h"    // VINS parameters (where CameraPtr is defined)

// If CameraPtr is not globally typedef'd in your project, define it or include proper header
// typedef std::shared_ptr<Camera> CameraPtr; 

bool SolvePoseWithMonoDepth(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const cv::Mat& depth_map1,
    SplgInference* matcher,
    CameraPtr camera,
    Eigen::Matrix3d& R_est,
    Eigen::Vector3d& t_est
);