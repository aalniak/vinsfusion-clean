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

// Base helper to share data setup logic
struct PhotometricRegBase {
    cv::Mat img_src_cpu;           
    cv::Mat depth_src_cpu;         
    cv::cuda::GpuMat img_tgt_gpu;  
    
    Eigen::Matrix3d K;
    double weight;
    double ssim_weight;
    double l1_weight;

    PhotometricRegBase(const cv::Mat& img_src, const cv::Mat& img_tgt, 
                       const cv::Mat& depth_src, const Eigen::Matrix3d& _K, 
                       double _w, double _ssim_w, double _l1_w) 
         : K(_K), weight(_w), ssim_weight(_ssim_w), l1_weight(_l1_w) {
         
         float scale_factor = 120.0f / img_src.cols;
         if (scale_factor < 1.0f) {
             cv::Mat img_src_small, depth_src_small, img_tgt_small;
             cv::resize(img_src, img_src_small, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
             cv::resize(depth_src, depth_src_small, cv::Size(), scale_factor, scale_factor, cv::INTER_NEAREST);
             cv::resize(img_tgt, img_tgt_small, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
             
             img_src_cpu = img_src_small;
             depth_src_cpu = depth_src_small;
             img_tgt_gpu.upload(img_tgt_small);
             
             K = K * scale_factor;
             K(2,2) = 1.0; 
         } else {
             img_src_cpu = img_src.clone();
             depth_src_cpu = depth_src.clone();
             img_tgt_gpu.upload(img_tgt);
         }
    }
    
    // Helper to compute T and Loss
    double check(const Eigen::Vector3d& P_src, const Eigen::Quaterniond& Q_src,
                 const Eigen::Vector3d& P_tgt, const Eigen::Quaterniond& Q_tgt,
                 double alpha, double beta, double scale, double shift) const {
                 
        Eigen::Matrix3d R_src = Q_src.toRotationMatrix();
        Eigen::Matrix3d R_tgt = Q_tgt.toRotationMatrix();
        Eigen::Matrix3d R_tgt_T = R_tgt.transpose();
        Eigen::Matrix3d R_rel = R_tgt_T * R_src;
        Eigen::Vector3d t_rel = R_tgt_T * (P_src - P_tgt);
        
        Eigen::Matrix4d T_tgt_src = Eigen::Matrix4d::Identity();
        T_tgt_src.block<3,3>(0,0) = R_rel;
        T_tgt_src.block<3,1>(0,3) = t_rel;
        
        return PhotometricLoss::computeGPU(
             img_src_cpu, img_tgt_gpu, depth_src_cpu, 
             K, T_tgt_src, ssim_weight, l1_weight,
             alpha, beta, scale, shift);
    }
};

// MODE 1: POSE ONLY (Standard)
struct PhotometricRegFactor : public PhotometricRegBase {
    using PhotometricRegBase::PhotometricRegBase; // Inherit Constructor

    bool operator()(const double* const pose_src, const double* const pose_tgt, double* residuals) const {
         Eigen::Vector3d P_src(pose_src[0], pose_src[1], pose_src[2]);
         Eigen::Quaterniond Q_src(pose_src[6], pose_src[3], pose_src[4], pose_src[5]);
         Eigen::Vector3d P_tgt(pose_tgt[0], pose_tgt[1], pose_tgt[2]);
         Eigen::Quaterniond Q_tgt(pose_tgt[6], pose_tgt[3], pose_tgt[4], pose_tgt[5]);
         
         // Defaults for others
         residuals[0] = weight * check(P_src, Q_src, P_tgt, Q_tgt, 1.0, 0.0, 1.0, 0.0);
         return true;
    }
    
    static ceres::CostFunction* Create(
        const cv::Mat& img_src, const cv::Mat& img_tgt, 
        const cv::Mat& depth_src, const Eigen::Matrix3d& K, 
        double weight, double ssim_w, double l1_w) 
    {
        return new ceres::NumericDiffCostFunction<PhotometricRegFactor, ceres::FORWARD, 1, 7, 7>(
            new PhotometricRegFactor(img_src, img_tgt, depth_src, K, weight, ssim_w, l1_w));
    }
};

// MODE 2: POSE + AFFINE
struct PhotometricRegFactorAffine : public PhotometricRegBase {
    using PhotometricRegBase::PhotometricRegBase;

    bool operator()(const double* const pose_src, const double* const pose_tgt, 
                    const double* const affine, double* residuals) const {
         Eigen::Vector3d P_src(pose_src[0], pose_src[1], pose_src[2]);
         Eigen::Quaterniond Q_src(pose_src[6], pose_src[3], pose_src[4], pose_src[5]);
         Eigen::Vector3d P_tgt(pose_tgt[0], pose_tgt[1], pose_tgt[2]);
         Eigen::Quaterniond Q_tgt(pose_tgt[6], pose_tgt[3], pose_tgt[4], pose_tgt[5]);
         
         double alpha = affine[0];
         double beta = affine[1];
         
         residuals[0] = weight * check(P_src, Q_src, P_tgt, Q_tgt, alpha, beta, 1.0, 0.0);
         return true;
    }
    
    static ceres::CostFunction* Create(
        const cv::Mat& img_src, const cv::Mat& img_tgt, 
        const cv::Mat& depth_src, const Eigen::Matrix3d& K, 
        double weight, double ssim_w, double l1_w) 
    {
        return new ceres::NumericDiffCostFunction<PhotometricRegFactorAffine, ceres::FORWARD, 1, 7, 7, 2>(
            new PhotometricRegFactorAffine(img_src, img_tgt, depth_src, K, weight, ssim_w, l1_w));
    }
};

// MODE 3: POSE + SCALE/SHIFT
struct PhotometricRegFactorScaleShift : public PhotometricRegBase {
    using PhotometricRegBase::PhotometricRegBase;

    bool operator()(const double* const pose_src, const double* const pose_tgt, 
                    const double* const scale_shift, double* residuals) const {
         Eigen::Vector3d P_src(pose_src[0], pose_src[1], pose_src[2]);
         Eigen::Quaterniond Q_src(pose_src[6], pose_src[3], pose_src[4], pose_src[5]);
         Eigen::Vector3d P_tgt(pose_tgt[0], pose_tgt[1], pose_tgt[2]);
         Eigen::Quaterniond Q_tgt(pose_tgt[6], pose_tgt[3], pose_tgt[4], pose_tgt[5]);
         
         double scale = scale_shift[0];
         double shift = scale_shift[1];
         
         residuals[0] = weight * check(P_src, Q_src, P_tgt, Q_tgt, 1.0, 0.0, scale, shift);
         return true;
    }
    
    static ceres::CostFunction* Create(
        const cv::Mat& img_src, const cv::Mat& img_tgt, 
        const cv::Mat& depth_src, const Eigen::Matrix3d& K, 
        double weight, double ssim_w, double l1_w) 
    {
        return new ceres::NumericDiffCostFunction<PhotometricRegFactorScaleShift, ceres::FORWARD, 1, 7, 7, 2>(
            new PhotometricRegFactorScaleShift(img_src, img_tgt, depth_src, K, weight, ssim_w, l1_w));
    }
};

}  // namespace vins::estimator
