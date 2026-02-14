#pragma once

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace vins::estimator {

// Interpolator Type Definition for convenience
// ColumnMajorAdapter for OpenCV Mat (RowMajor) requires care, but Grid2D handles RowMajorData if we specify it.
// Actually Ceres Grid2D defaults to RowMajor for "DATA_DIMENSION = 1" (Scalar).
// RowMajor: data[r * cols + c]
typedef ceres::Grid2D<unsigned char, 1> ImageGrid;
typedef ceres::BiCubicInterpolator<ImageGrid> ImageInterpolator;

struct PhotometricSparseFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Measurements
    Eigen::Vector2d pt_ref; // Normalized coordinates (u, v)
    double intensity_ref;
    double inv_depth_ref;
    
    // Intrinsics
    Eigen::Matrix3d K;
    
    // Interpolator for Target Image
    const ImageInterpolator* interpolator;
    
    // Weights
    double weight;
    
    // Mode flags (compile time would be better, but runtime is fine for now)
    bool use_affine;
    bool use_scale_shift;

    PhotometricSparseFactor(const Eigen::Vector2d& _pt_ref, double _intensity, double _inv_depth,
                           const Eigen::Matrix3d& _K, const ImageInterpolator* _interp, double _weight,
                           bool _affine = false, bool _ss = false)
        : pt_ref(_pt_ref), intensity_ref(_intensity), inv_depth_ref(_inv_depth),
          K(_K), interpolator(_interp), weight(_weight),
          use_affine(_affine), use_scale_shift(_ss) {}

    template <typename T>
    bool operator()(const T* const pose_cur, const T* const param_affine, const T* const param_ss, T* residuals) const {
        // Unpack Pose (Cur w.r.t Ref? No, Global Poses usually)
        // In PhotometricRefinement threadLoop, we set Ref pose to Identity and optimize Cur.
        // So pose_cur is T_ref_cur (Pose of Cur in Ref frame).
        // Let's assume standard convention:
        // P_cur = R_ref_cur * P_ref + t_ref_cur ? No via Ref->Cur.
        // T_ref_cur usually means: Points in Cur = T * Points in Ref.
        
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_cur(pose_cur);
        Eigen::Map<const Eigen::Quaternion<T>> q_cur(pose_cur + 3); // stored as [x, y, z, w] in Ceres map? No, usually [x, y, z] + [w, x, y, z]
        // Ceres Quaternion is [w, x, y, z] usually? Wait.
        // In VINS estimator.cpp: para_Pose[i][6] = q.w().
        // Ceres uses [w, x, y, z] for EigenQuaternionParameterization BUT
        // Eigen::Quaternion memory layout is [x, y, z, w].
        // vins uses `ceres::EigenQuaternionManifold` (or custom).
        // If we use EigenQuaternionManifold, the pointer expects [x, y, z, w].
        // So Map<Quaternion> works correctly as it expects [x, y, z, w].
        
        // T_cur_ref (Ref -> Cur)
        // P_cur = q_cur * P_ref + t_cur
        
        // 1. Back-project Ref Point
        // Apply Scale/Shift if enabled
        T depth = T(1.0) / T(inv_depth_ref);
        
        if (use_scale_shift) {
            T s = param_ss[0];
            T h = param_ss[1];
            T inv_d_metric = s * T(inv_depth_ref) + h;
            depth = T(1.0) / inv_d_metric;
        }
        
        T fx = T(K(0,0));
        T fy = T(K(1,1));
        T cx = T(K(0,2));
        T cy = T(K(1,2));
        
        T x_n = (T(pt_ref.x()) - cx) / fx;
        T y_n = (T(pt_ref.y()) - cy) / fy;
        
        Eigen::Matrix<T, 3, 1> P_ref;
        P_ref << x_n * depth, y_n * depth, depth;
        
        // 2. Transform to Cur
        Eigen::Matrix<T, 3, 1> P_cur = q_cur * P_ref + t_cur;
        
        // 3. Project to Cur Image
        T u_cur = fx * P_cur.x() / P_cur.z() + cx;
        T v_cur = fy * P_cur.y() / P_cur.z() + cy;
        
        // 4. Interpolate
        T intensity_cur;
        interpolator->Evaluate(v_cur, u_cur, &intensity_cur); // Grid is (row, col) -> (y, x)
        
        // 5. Residual
        // I_cur - (alpha * I_ref + betaI)
        T pred_intensity = intensity_cur;
        T target_intensity = T(intensity_ref);
        
        if (use_affine) {
            T alpha = param_affine[0];
            T beta = param_affine[1];
            target_intensity = alpha * target_intensity + beta;
        }
        
        residuals[0] = T(weight) * (pred_intensity - target_intensity);
        
        return true;
    }
    
    // Factory for different modes to simplify creation
};

// Mode Wrappers to handle different parameter block counts

struct PhotometricSparseFactorPose {
    PhotometricSparseFactor factor;
    PhotometricSparseFactorPose(const PhotometricSparseFactor& f) : factor(f) {}
    
    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        T dummy_affine[2] = {T(1), T(0)};
        T dummy_ss[2] = {T(1), T(0)};
        return factor(pose, dummy_affine, dummy_ss, residuals);
    }
};

struct PhotometricSparseFactorAffine {
    PhotometricSparseFactor factor;
    PhotometricSparseFactorAffine(const PhotometricSparseFactor& f) : factor(f) {}
    
    template <typename T>
    bool operator()(const T* const pose, const T* const affine, T* residuals) const {
        T dummy_ss[2] = {T(1), T(0)};
        return factor(pose, affine, dummy_ss, residuals);
    }
};

struct PhotometricSparseFactorScaleShift {
    PhotometricSparseFactor factor;
    PhotometricSparseFactorScaleShift(const PhotometricSparseFactor& f) : factor(f) {}
    
    template <typename T>
    bool operator()(const T* const pose, const T* const ss, T* residuals) const {
        T dummy_affine[2] = {T(1), T(0)};
        return factor(pose, dummy_affine, ss, residuals);
    }
};

struct PhotometricSparseFactorAffineScaleShift {
    PhotometricSparseFactor factor;
    PhotometricSparseFactorAffineScaleShift(const PhotometricSparseFactor& f) : factor(f) {}
    
    template <typename T>
    bool operator()(const T* const pose, const T* const affine, const T* const ss, T* residuals) const {
        return factor(pose, affine, ss, residuals);
    }
};

// Regularization: penalize affine [alpha, beta] deviating from [1, 0]
struct AffineRegularization {
    double weight_alpha;
    double weight_beta;
    
    AffineRegularization(double wa, double wb) : weight_alpha(wa), weight_beta(wb) {}
    
    template <typename T>
    bool operator()(const T* const affine, T* residuals) const {
        residuals[0] = T(weight_alpha) * (affine[0] - T(1.0)); // alpha should be 1
        residuals[1] = T(weight_beta) * affine[1];              // beta should be 0
        return true;
    }
};

// Regularization: penalize scale_shift [s, h] deviating from [1, 0]
struct ScaleShiftRegularization {
    double weight_scale;
    double weight_shift;
    
    ScaleShiftRegularization(double ws, double wh) : weight_scale(ws), weight_shift(wh) {}
    
    template <typename T>
    bool operator()(const T* const ss, T* residuals) const {
        residuals[0] = T(weight_scale) * (ss[0] - T(1.0)); // scale should be 1
        residuals[1] = T(weight_shift) * ss[1];              // shift should be 0
        return true;
    }
};

} // namespace vins::estimator
