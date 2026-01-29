/*******************************************************
 * Fused Depth Prior Factor for VINS-Fusion
 * 
 * Uses Bayesian-fused depth from multiple viewpoints as a more
 * stable depth prior than single-frame predictions. The fusion
 * averages out random flickering in zero-shot depth models.
 * 
 * Key features:
 * - Scale-invariant residual: (d_vio / d_prior) - 1
 * - Uncertainty-aware: weight = 1/σ_fused
 * - Only activates when enough viewpoints have been observed
 *******************************************************/

#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>

namespace vins::estimator {

/**
 * FusedDepthPriorFactor
 * 
 * Pulls VIO inverse-depth toward the Bayesian-fused depth prediction.
 * Uses scale-invariant residual for robustness to systematic scale errors.
 * 
 * Residual = sqrt_info * ((inv_depth_vio / fused_inv_depth) - 1)
 * 
 * This residual is scale-invariant: multiplying both depths by any constant
 * leaves the residual unchanged.
 */
struct FusedDepthPriorFactor
{
    const double fused_inv_depth;  // Bayesian-fused inverse depth
    const double sqrt_info;        // sqrt(1/variance) = information

    FusedDepthPriorFactor(double fused_inv_d, double fused_variance)
        : fused_inv_depth(fused_inv_d), 
          sqrt_info(1.0 / std::sqrt(fused_variance + 1e-8)) {}


    template <typename T>
    bool operator()(const T* const inv_depth_vio, T* residuals) const
    {
        // Scale-invariant residual: (d_vio / d_prior) - 1
        // When d_vio ≈ d_prior, residual ≈ 0
        // When d_vio = 2 * d_prior, residual = 1 (100% error)
        residuals[0] = T(sqrt_info) * (inv_depth_vio[0] / T(fused_inv_depth) - T(1.0));
        return true;
    }

    static ceres::CostFunction* Create(double fused_inv_d, double fused_variance) {
        return new ceres::AutoDiffCostFunction<FusedDepthPriorFactor, 1, 1>(
            new FusedDepthPriorFactor(fused_inv_d, fused_variance));
    }
};

/**
 * WeightedFusedDepthFactor
 * 
 * Same as FusedDepthPriorFactor but with an additional external weight
 * multiplier (e.g., from parallax-based gating).
 */
struct WeightedFusedDepthFactor
{
    const double fused_inv_depth;
    const double sqrt_info;
    const double external_weight;

    WeightedFusedDepthFactor(double fused_inv_d, double fused_variance, double ext_weight)
        : fused_inv_depth(fused_inv_d), 
          sqrt_info(1.0 / std::sqrt(fused_variance + 1e-8)),
          external_weight(ext_weight) {}

    template <typename T>
    bool operator()(const T* const inv_depth_vio, T* residuals) const
    {
        // Absolute residual scaled by Bayesian info and external weight
        residuals[0] = T(sqrt_info * external_weight) * 
                       (inv_depth_vio[0] - T(fused_inv_depth));
        return true;
    }

    static ceres::CostFunction* Create(double fused_inv_d, double fused_variance, double ext_weight) {
        return new ceres::AutoDiffCostFunction<WeightedFusedDepthFactor, 1, 1>(
            new WeightedFusedDepthFactor(fused_inv_d, fused_variance, ext_weight));
    }
};

}  // namespace vins::estimator
