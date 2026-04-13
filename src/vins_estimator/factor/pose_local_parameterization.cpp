/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vins_estimator/factor/pose_local_parameterization.h>

namespace vins::estimator {

bool PoseManifold::Plus(const double *x, const double *delta,
                        double *x_plus_delta) const {
  Eigen::Map<const Eigen::Vector3d> _p(x);
  Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

  Eigen::Map<const Eigen::Vector3d> dp(delta);

  Eigen::Quaterniond dq =
      Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

  Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
  Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

  p = _p + dp;
  q = (_q * dq).normalized();

  return true;
}
bool PoseManifold::PlusJacobian(const double *x, double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
  j.topRows<6>().setIdentity();
  j.bottomRows<1>().setZero();

  return true;
}

bool PoseManifold::Minus(const double *y, const double *x,
                         double *y_minus_x) const {
  // Position: simple subtraction
  Eigen::Map<const Eigen::Vector3d> p_y(y);
  Eigen::Map<const Eigen::Vector3d> p_x(x);
  y_minus_x[0] = p_y.x() - p_x.x();
  y_minus_x[1] = p_y.y() - p_x.y();
  y_minus_x[2] = p_y.z() - p_x.z();

  // Rotation: find delta_theta s.t. q_x * deltaQ(delta_theta) = q_y
  // => deltaQ(delta_theta) = q_x^{-1} * q_y
  Eigen::Map<const Eigen::Quaterniond> q_y(y + 3);
  Eigen::Map<const Eigen::Quaterniond> q_x(x + 3);
  Eigen::Quaterniond dq = q_x.inverse() * q_y;
  // Ensure shorter path
  if (dq.w() < 0.0) dq.coeffs() *= -1.0;

  // Consistent with deltaQ: dq ≈ [theta/2, 1] => theta = 2 * dq.vec()
  // For large angles use atan2 for numerical accuracy
  double sin_half = dq.vec().norm();
  if (sin_half < 1e-10) {
    y_minus_x[3] = 2.0 * dq.x();
    y_minus_x[4] = 2.0 * dq.y();
    y_minus_x[5] = 2.0 * dq.z();
  } else {
    double half_angle = std::atan2(sin_half, dq.w());
    double scale = 2.0 * half_angle / sin_half;
    y_minus_x[3] = scale * dq.x();
    y_minus_x[4] = scale * dq.y();
    y_minus_x[5] = scale * dq.z();
  }
  return true;
}

bool PoseManifold::MinusJacobian(const double *x, double *jacobian) const {
  // MinusJacobian: d(Minus(y, x)) / d(y) evaluated at y = x
  // Result is TangentSize x AmbientSize = 6 x 7
  Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> j(jacobian);
  j.setZero();

  // Position part: d(p_y - p_x)/d(p_y) = I_3
  j.block<3, 3>(0, 0).setIdentity();

  // Rotation part: d(2 * vec(q_x^{-1} * q_y))/d(q_y) at q_y = q_x
  // q_x^{-1} * q_y is a left-multiplication by q_x^{-1} which is linear in q_y.
  // The Jacobian is 2 * [vector rows of L(q_x^{-1})], a 3x4 matrix.
  // L(q) * p gives the quaternion product q*p in [x,y,z,w] storage.
  Eigen::Map<const Eigen::Quaterniond> q(x + 3);
  // q^{-1} for unit quaternion: [-qx, -qy, -qz, qw]
  double qx = -q.x(), qy = -q.y(), qz = -q.z(), qw = q.w();

  // L(q^{-1}) vector rows (3x4 in [x,y,z,w] column order):
  // row_x: [ qw, -qz,  qy, qx]
  // row_y: [ qz,  qw, -qx, qy]
  // row_z: [-qy,  qx,  qw, qz]
  // Cols map to q_y components: [qy_x, qy_y, qy_z, qy_w] at indices [3,4,5,6]
  j(3, 3) =  qw;  j(3, 4) = -qz;  j(3, 5) =  qy;  j(3, 6) = qx;
  j(4, 3) =  qz;  j(4, 4) =  qw;  j(4, 5) = -qx;  j(4, 6) = qy;
  j(5, 3) = -qy;  j(5, 4) =  qx;  j(5, 5) =  qw;  j(5, 6) = qz;
  // Scale by 2 for the theta = 2*vec(dq) convention
  j.block<3, 4>(3, 3) *= 2.0;

  return true;
}

}  // namespace vins::estimator