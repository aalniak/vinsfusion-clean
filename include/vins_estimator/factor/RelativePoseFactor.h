#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <vins_estimator/utility/utility.h>

namespace vins::estimator {

class RelativePoseFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RelativePoseFactor(const Eigen::Vector3d& t_meas, const Eigen::Quaterniond& q_meas, const Eigen::Matrix<double, 6, 6>& sqrt_info)
        : t_meas_(t_meas), q_meas_(q_meas), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pi(pose_i);
        Eigen::Map<const Eigen::Quaternion<T>> Qi(pose_i + 3);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pj(pose_j);
        Eigen::Map<const Eigen::Quaternion<T>> Qj(pose_j + 3);

        // Relative transform: T_ij = T_i^{-1} * T_j
        Eigen::Quaternion<T> Qi_inv = Qi.inverse();
        Eigen::Matrix<T, 3, 1> t_est = Qi_inv * (Pj - Pi);
        Eigen::Quaternion<T> q_est = Qi_inv * Qj;

        // Error
        Eigen::Matrix<T, 3, 1> t_err = t_est - t_meas_.cast<T>();
        
        Eigen::Quaternion<T> q_meas_cast = q_meas_.cast<T>();
        Eigen::Quaternion<T> q_err = q_meas_cast.inverse() * q_est;
        
        // Normalize q_err to ensure valid vector part extraction
        q_err.normalize();
        
        Eigen::Matrix<T, 3, 1> theta_err = T(2.0) * q_err.vec();
        
        // Residuals
        Eigen::Matrix<T, 6, 1> raw_residual;
        raw_residual.head(3) = t_err;
        raw_residual.tail(3) = theta_err;

        // Weighting
        // sqrt_info is 6x6. 
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residual_map(residuals);
        residual_map = sqrt_info_.cast<T>() * raw_residual;

        // DEBUG: Verify it runs
        static int debug_cnt = 0;
        if (debug_cnt++ % 1000 == 0) {
            // checking residual mag
            // printf("[RelativePoseFactor] Eval. t_err norm: %f\n", (float)t_err.norm());
        }

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& t_meas, 
                                       const Eigen::Quaterniond& q_meas, 
                                       const Eigen::Matrix<double, 6, 6>& sqrt_info) {
        return new ceres::AutoDiffCostFunction<RelativePoseFactor, 6, 7, 7>(
            new RelativePoseFactor(t_meas, q_meas, sqrt_info));
    }

private:
    Eigen::Vector3d t_meas_;
    Eigen::Quaterniond q_meas_;
    Eigen::Matrix<double, 6, 6> sqrt_info_;
};

} // namespace vins::estimator
