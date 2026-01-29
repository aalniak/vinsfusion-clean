/*******************************************************
 * Photometric Regularization Factor
 * 
 * Wraps PhotometricLoss utility in a Ceres CostFunction.
 * Uses Numeric Differentiation for simplicity as analytical 
 * Jacobians for SSIM+ImageWarping are complex.
 * 
 * Optimizes the relative pose between two frames to minimize
 * photometric consistency error, guided by monocular depth.
 *******************************************************/

#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <opencv2/core/cuda.hpp>
#include <vins_estimator/utility/photometricLoss.h>

namespace vins::estimator {

struct PhotometricRegFactor {
    // Hybrid storage for optimal performance
    cv::Mat img_src_cpu;           // Reference image on CPU (for loss calc)
    cv::Mat depth_src_cpu;         // Depth on CPU (for map generation)
    cv::cuda::GpuMat img_tgt_gpu;  // Target image on GPU (source for warping)
    
    Eigen::Matrix3d K;
    double weight;
    double ssim_weight;
    double l1_weight;

    PhotometricRegFactor(const cv::Mat& img_src, const cv::Mat& img_tgt, 
                         const cv::Mat& depth_src, const Eigen::Matrix3d& _K, 
                         double _w, double _ssim_w, double _l1_w) 
         : K(_K), weight(_w), ssim_weight(_ssim_w), l1_weight(_l1_w) {
         
         // OPTIMIZATION: Downsample to speed up warping (target width ~120px)
         float scale_factor = 120.0f / img_src.cols;
         if (scale_factor < 1.0f) {
             cv::Mat img_src_small, depth_src_small, img_tgt_small;
             cv::resize(img_src, img_src_small, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
             cv::resize(depth_src, depth_src_small, cv::Size(), scale_factor, scale_factor, cv::INTER_NEAREST); // Depth needs nearest or care
             cv::resize(img_tgt, img_tgt_small, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
             
             img_src_cpu = img_src_small;
             depth_src_cpu = depth_src_small;
             img_tgt_gpu.upload(img_tgt_small);
             
             // Scale Intrinsics
             K = K * scale_factor;
             K(2,2) = 1.0; // Maintain homogenous scaling
         } else {
             // Store static reference data on CPU to avoid "ping-pong" transfers
             img_src_cpu = img_src.clone();
             depth_src_cpu = depth_src.clone();
             
             // Upload target image to GPU once (it stays static, only warp maps change)
             img_tgt_gpu.upload(img_tgt);
         }
    }

    bool operator()(const double* const pose_src, const double* const pose_tgt, double* residuals) const {
         // VINS Pose: [tx, ty, tz, qx, qy, qz, qw]
         Eigen::Vector3d P_src(pose_src[0], pose_src[1], pose_src[2]);
         Eigen::Quaterniond Q_src(pose_src[6], pose_src[3], pose_src[4], pose_src[5]);
         
         Eigen::Vector3d P_tgt(pose_tgt[0], pose_tgt[1], pose_tgt[2]);
         Eigen::Quaterniond Q_tgt(pose_tgt[6], pose_tgt[3], pose_tgt[4], pose_tgt[5]);
         
         // Compute relative pose T_tgt_src (transforms points from src to tgt)
         Eigen::Matrix3d R_src = Q_src.toRotationMatrix();
         Eigen::Matrix3d R_tgt = Q_tgt.toRotationMatrix();
         
         // T_w_tgt^-1 = [R_tgt^T, -R_tgt^T * P_tgt]
         Eigen::Matrix3d R_tgt_T = R_tgt.transpose();
         
         // T_relative = T_w_tgt^-1 * T_w_src
         Eigen::Matrix3d R_rel = R_tgt_T * R_src;
         Eigen::Vector3d t_rel = R_tgt_T * (P_src - P_tgt);
         
         Eigen::Matrix4d T_tgt_src = Eigen::Matrix4d::Identity();
         T_tgt_src.block<3,3>(0,0) = R_rel;
         T_tgt_src.block<3,1>(0,3) = t_rel;
         
         // Compute Loss using optimized Hybrid CPU/GPU method
         double loss = PhotometricLoss::computeGPU(
             img_src_cpu, img_tgt_gpu, depth_src_cpu, 
             K, T_tgt_src, ssim_weight, l1_weight);
         
         residuals[0] = weight * loss;
         return true;
    }
    
    // Factory method to create a NumericDiffCostFunction
    static ceres::CostFunction* Create(
        const cv::Mat& img_src, const cv::Mat& img_tgt, 
        const cv::Mat& depth_src, const Eigen::Matrix3d& K, 
        double weight, double ssim_w, double l1_w) 
    {
        // Use FORWARD difference for speed (14 evals instead of 28)
        // Residual dim: 1
        // Params: Pose_src (7), Pose_tgt (7)
        return new ceres::NumericDiffCostFunction<PhotometricRegFactor, ceres::FORWARD, 1, 7, 7>(
            new PhotometricRegFactor(img_src, img_tgt, depth_src, K, weight, ssim_w, l1_w));
    }
};

}  // namespace vins::estimator
