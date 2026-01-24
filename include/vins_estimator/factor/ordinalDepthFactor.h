/*******************************************************
 * Ordinal Depth Factor for VINS-Fusion
 * 
 * Enforces relative depth ordering between feature pairs based on
 * zero-shot monocular depth predictions. This is more robust than
 * absolute depth priors because relative ordering is preserved even
 * when the absolute scale/shift flickers.
 * 
 * Residual: max(0, margin - (inv_depth_closer - inv_depth_farther))
 * Only penalizes when the ordering constraint is violated.
 *******************************************************/

#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>

namespace vins::estimator {

/**
 * OrdinalDepthFactor
 * 
 * Enforces that the inverse depth of the "closer" feature is greater than
 * the "farther" feature by at least a margin.
 * 
 * In inverse depth space: closer objects have HIGHER inverse depth values.
 * If monocular depth says feature A is closer than B, we enforce:
 *   inv_depth_A >= inv_depth_B + margin
 * 
 * Residual = max(0, margin - (inv_depth_closer - inv_depth_farther))
 *          = max(0, margin + inv_depth_farther - inv_depth_closer)
 * 
 * This is a hinge loss - zero when constraint is satisfied, positive when violated.
 */
struct OrdinalDepthFactor
{
    const double margin;      // Minimum inv-depth difference to enforce
    const double sqrt_info;   // Weight (sqrt of information)

    OrdinalDepthFactor(double _margin, double _weight)
        : margin(_margin), sqrt_info(_weight) {}

    template <typename T>
    bool operator()(const T* const inv_depth_closer,   // Feature that should be closer (higher inv-depth)
                    const T* const inv_depth_farther,  // Feature that should be farther (lower inv-depth)
                    T* residuals) const
    {
        // Constraint: inv_depth_closer >= inv_depth_farther + margin
        // Rearranged: margin + inv_depth_farther - inv_depth_closer <= 0
        T violation = T(margin) + inv_depth_farther[0] - inv_depth_closer[0];
        
        // Hinge loss: only penalize when violated (violation > 0)
        // Using softplus for smooth gradient: log(1 + exp(k * violation)) / k
        // For simplicity, using max(0, violation) with a smooth approximation
        const T k = T(10.0);  // Sharpness of the soft-hinge
        residuals[0] = T(sqrt_info) * ceres::log(T(1.0) + ceres::exp(k * violation)) / k;
        
        return true;
    }

    static ceres::CostFunction* Create(double margin, double weight) {
        return new ceres::AutoDiffCostFunction<OrdinalDepthFactor, 1, 1, 1>(
            new OrdinalDepthFactor(margin, weight));
    }
};

/**
 * Utility struct to track ordinal depth pairs for a frame
 */
struct OrdinalPair {
    int feature_idx_closer;   // Index in para_Feature of closer feature
    int feature_idx_farther;  // Index in para_Feature of farther feature
    double inv_depth_diff;    // Difference in mono inv-depth (for sorting by confidence)
};

}  // namespace vins::estimator
