/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ceres/ceres.h>

namespace vins::estimator {

class TranslationNormFactor {
 public:
  TranslationNormFactor(double target_norm, double weight)
      : target_norm_(target_norm), weight_(weight) {}

  template <typename T>
  bool operator()(const T *const pose_i, const T *const pose_j,
                  T *residuals) const {
    const T dx = pose_j[0] - pose_i[0];
    const T dy = pose_j[1] - pose_i[1];
    const T dz = pose_j[2] - pose_i[2];
    const T norm = ceres::sqrt(dx * dx + dy * dy + dz * dz + T(1e-12));
    residuals[0] = T(weight_) * (norm - T(target_norm_));
    return true;
  }

  static ceres::CostFunction *Create(double target_norm, double weight) {
    return new ceres::AutoDiffCostFunction<TranslationNormFactor, 1, 7, 7>(
        new TranslationNormFactor(target_norm, weight));
  }

 private:
  double target_norm_;
  double weight_;
};

}  // namespace vins::estimator
