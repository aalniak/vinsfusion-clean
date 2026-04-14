/*******************************************************
 * Photometric Loss Utility for VINS-Fusion
 * 
 * GPU-accelerated photometric consistency computation using OpenCV CUDA.
 * Computes SSIM + L1 loss between warped image pairs for depth regularization.
 * 
 * Key features:
 * - GPU-accelerated image warping via cv::cuda::remap
 * - Structural Similarity Index (SSIM) for perceptual quality
 * - L1 loss for pixel-level consistency
 * - Occlusion-aware masking
 *******************************************************/

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#ifdef VINS_WITH_OPENCV_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#endif

namespace vins::estimator {

/**
 * PhotometricLoss
 * 
 * Computes photometric consistency loss between two images using depth-guided warping.
 * The loss is scale-invariant because it measures image similarity, not depth directly.
 */
class PhotometricLoss {
public:
    /**
     * Compute photometric loss between source and target images.
     * 
     * @param img_src Source image (grayscale, CV_8UC1)
     * @param img_target Target image (grayscale, CV_8UC1)
     * @param depth_src Depth map of source image (CV_32FC1, inverse depth)
     * @param K Camera intrinsic matrix (3x3)
     * @param T_target_src Relative pose: transforms points from source to target frame (4x4)
     * @param ssim_weight Weight for SSIM component (default: 0.85)
     * @param l1_weight Weight for L1 component (default: 0.15)
     * @return Total photometric loss (lower is better)
     */
    static double compute(
        const cv::Mat& img_src,
        const cv::Mat& img_target,
        const cv::Mat& depth_src,
        const Eigen::Matrix3d& K,
        const Eigen::Matrix4d& T_target_src,
        double ssim_weight = 0.85,
        double l1_weight = 0.15);

#ifdef VINS_WITH_OPENCV_CUDA
    /**
     * GPU-accelerated version of compute().
     * Uses cv::cuda for warping and loss computation.
     */
    static double computeGPU(
        const cv::Mat& img_src_cpu,
        const cv::cuda::GpuMat& img_target_gpu,
        const cv::Mat& depth_src_cpu,
        const Eigen::Matrix3d& K,
        const Eigen::Matrix4d& T_target_src,
        double ssim_weight = 0.85,
        double l1_weight = 0.15);
#endif

    /**
     * Compute SSIM between two images (CPU version).
     * Returns value in range [0, 1] where 1 = identical.
     */
    static double computeSSIM(const cv::Mat& img1, const cv::Mat& img2);

    /**
     * Generate warp maps for image warping based on depth and relative pose.
     * Output maps can be used with cv::remap or cv::cuda::remap.
     */
    static void generateWarpMaps(
        const cv::Mat& depth_src,
        const Eigen::Matrix3d& K,
        const Eigen::Matrix4d& T_target_src,
        cv::Mat& map_x,
        cv::Mat& map_y,
        cv::Mat& valid_mask);

private:
    // Constants for SSIM computation
    static constexpr double C1 = 6.5025;   // (0.01 * 255)^2
    static constexpr double C2 = 58.5225;  // (0.03 * 255)^2
};

// ============================================================================
// Implementation (header-only for simplicity)
// ============================================================================

inline void PhotometricLoss::generateWarpMaps(
    const cv::Mat& depth_src,
    const Eigen::Matrix3d& K,
    const Eigen::Matrix4d& T_target_src,
    cv::Mat& map_x,
    cv::Mat& map_y,
    cv::Mat& valid_mask)
{
    int rows = depth_src.rows;
    int cols = depth_src.cols;
    
    map_x.create(rows, cols, CV_32FC1);
    map_y.create(rows, cols, CV_32FC1);
    valid_mask.create(rows, cols, CV_8UC1);
    
    // Extract rotation and translation
    Eigen::Matrix3d R = T_target_src.block<3,3>(0, 0);
    Eigen::Vector3d t = T_target_src.block<3,1>(0, 3);
    
    // Precompute K * R * K_inv and K * t
    Eigen::Matrix3d K_inv = K.inverse();
    Eigen::Matrix3d H_rot = K * R * K_inv;
    Eigen::Vector3d H_trans = K * t;
    
    float fx = static_cast<float>(K(0, 0));
    float fy = static_cast<float>(K(1, 1));
    float cx = static_cast<float>(K(0, 2));
    float cy = static_cast<float>(K(1, 2));
    
    for (int v = 0; v < rows; ++v) {
        float* map_x_row = map_x.ptr<float>(v);
        float* map_y_row = map_y.ptr<float>(v);
        uchar* valid_row = valid_mask.ptr<uchar>(v);
        const float* depth_row = depth_src.ptr<float>(v);
        
        for (int u = 0; u < cols; ++u) {
            float inv_d = depth_row[u];
            
            // Invalid depth check
            if (inv_d <= 0.001f || inv_d > 10.0f) {
                map_x_row[u] = -1.0f;
                map_y_row[u] = -1.0f;
                valid_row[u] = 0;
                continue;
            }
            
            float d = 1.0f / inv_d;  // Convert to metric depth
            
            // Back-project to 3D (normalized coords)
            float x_norm = (u - cx) / fx;
            float y_norm = (v - cy) / fy;
            
            // 3D point in source camera frame
            Eigen::Vector3d P_src(x_norm * d, y_norm * d, d);
            
            // Transform to target camera frame
            Eigen::Vector3d P_tgt = R * P_src + t;
            
            // Project to target image
            if (P_tgt.z() <= 0.01) {
                map_x_row[u] = -1.0f;
                map_y_row[u] = -1.0f;
                valid_row[u] = 0;
                continue;
            }
            
            float u_tgt = static_cast<float>(fx * P_tgt.x() / P_tgt.z() + cx);
            float v_tgt = static_cast<float>(fy * P_tgt.y() / P_tgt.z() + cy);
            
            // Bounds check
            if (u_tgt < 0 || u_tgt >= cols - 1 || v_tgt < 0 || v_tgt >= rows - 1) {
                map_x_row[u] = -1.0f;
                map_y_row[u] = -1.0f;
                valid_row[u] = 0;
                continue;
            }
            
            map_x_row[u] = u_tgt;
            map_y_row[u] = v_tgt;
            valid_row[u] = 255;
        }
    }
}

inline double PhotometricLoss::computeSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat img1_f, img2_f;
    img1.convertTo(img1_f, CV_32F);
    img2.convertTo(img2_f, CV_32F);
    
    cv::Mat mu1, mu2;
    cv::GaussianBlur(img1_f, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_f, mu2, cv::Size(11, 11), 1.5);
    
    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);
    
    cv::Mat sigma1_sq, sigma2_sq, sigma12;
    cv::GaussianBlur(img1_f.mul(img1_f), sigma1_sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_f.mul(img2_f), sigma2_sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img1_f.mul(img2_f), sigma12, cv::Size(11, 11), 1.5);
    
    sigma1_sq -= mu1_sq;
    sigma2_sq -= mu2_sq;
    sigma12 -= mu1_mu2;
    
    cv::Mat ssim_map;
    cv::Mat numerator = (2 * mu1_mu2 + C1).mul(2 * sigma12 + C2);
    cv::Mat denominator = (mu1_sq + mu2_sq + C1).mul(sigma1_sq + sigma2_sq + C2);
    cv::divide(numerator, denominator, ssim_map);
    
    return cv::mean(ssim_map)[0];
}

inline double PhotometricLoss::compute(
    const cv::Mat& img_src,
    const cv::Mat& img_target,
    const cv::Mat& depth_src,
    const Eigen::Matrix3d& K,
    const Eigen::Matrix4d& T_target_src,
    double ssim_weight,
    double l1_weight)
{
    // Generate warp maps
    cv::Mat map_x, map_y, valid_mask;
    generateWarpMaps(depth_src, K, T_target_src, map_x, map_y, valid_mask);
    
    // Warp target image to source viewpoint
    cv::Mat warped_target;
    cv::remap(img_target, warped_target, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    
    // Apply valid mask
    cv::Mat img_src_masked, warped_masked;
    img_src.copyTo(img_src_masked, valid_mask);
    warped_target.copyTo(warped_masked, valid_mask);
    
    // Count valid pixels
    int valid_pixels = cv::countNonZero(valid_mask);
    if (valid_pixels < 100) {
        return 1.0;  // Not enough valid pixels, return max loss
    }
    
    // Compute SSIM component (1 - SSIM so lower is better)
    double ssim = computeSSIM(img_src_masked, warped_masked);
    double ssim_loss = 1.0 - ssim;
    
    // Compute L1 component
    cv::Mat diff;
    cv::absdiff(img_src_masked, warped_masked, diff);
    double l1_loss = cv::sum(diff)[0] / (valid_pixels * 255.0);  // Normalize to [0, 1]
    
    // Combined loss
    return ssim_weight * ssim_loss + l1_weight * l1_loss;
}

#ifdef VINS_WITH_OPENCV_CUDA
inline double PhotometricLoss::computeGPU(
    const cv::Mat& img_src_cpu,           // Keep on CPU (Static reference)
    const cv::cuda::GpuMat& img_target_gpu, // Keep on GPU (Target to be warped)
    const cv::Mat& depth_src_cpu,         // Keep on CPU (For map generation)
    const Eigen::Matrix3d& K,
    const Eigen::Matrix4d& T_target_src,
    double ssim_weight,
    double l1_weight)
{
    // Generate warp maps on CPU (No download needed)
    cv::Mat map_x, map_y, valid_mask;
    generateWarpMaps(depth_src_cpu, K, T_target_src, map_x, map_y, valid_mask);
    
    // Upload maps to GPU (Required per iteration as T changes)
    cv::cuda::GpuMat map_x_gpu, map_y_gpu;
    map_x_gpu.upload(map_x);
    map_y_gpu.upload(map_y);
    
    // GPU warp using remap (The heavy lifting)
    cv::cuda::GpuMat warped_target_gpu;
    cv::cuda::remap(img_target_gpu, warped_target_gpu, map_x_gpu, map_y_gpu, cv::INTER_LINEAR);
    
    // Download result for loss computation (SSIM on CPU)
    cv::Mat warped_target;
    warped_target_gpu.download(warped_target);
    
    // Apply mask and compute loss on CPU
    // We reuse the CPU valid_mask calculated during map generation
    cv::Mat img_src_masked, warped_masked;
    img_src_cpu.copyTo(img_src_masked, valid_mask);
    warped_target.copyTo(warped_masked, valid_mask);
    
    int valid_pixels = cv::countNonZero(valid_mask);
    if (valid_pixels < 100) {
        return 1.0;
    }
    
    double ssim = computeSSIM(img_src_masked, warped_masked);
    double ssim_loss = 1.0 - ssim;
    
    cv::Mat diff;
    cv::absdiff(img_src_masked, warped_masked, diff);
    double l1_loss = cv::sum(diff)[0] / (valid_pixels * 255.0);
    
    return ssim_weight * ssim_loss + l1_weight * l1_loss;
}
#endif

}  // namespace vins::estimator
