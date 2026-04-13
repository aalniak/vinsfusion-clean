/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <algorithm>
#include <list>
#include <numeric>
#include <vector>
using namespace std;
#include <fstream>
#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/assert.h>
#include <ros/console.h>
#include <vins_estimator/estimator/parameters.h>
#include <vins_estimator/utility/tic_toc.h>

namespace vins::estimator {

class FeaturePerFrame {
 public:
  FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td) {
    point.x() = _point(0);
    point.y() = _point(1);
    point.z() = _point(2);
    uv.x() = _point(3);
    uv.y() = _point(4);
    velocity.x() = _point(5);
    velocity.y() = _point(6);
    cur_td = td;
    is_stereo = false;
  }

  void rightObservation(const Eigen::Matrix<double, 7, 1> &_point) {
    pointRight.x() = _point(0);
    pointRight.y() = _point(1);
    pointRight.z() = _point(2);
    uvRight.x() = _point(3);
    uvRight.y() = _point(4);
    velocityRight.x() = _point(5);
    velocityRight.y() = _point(6);
    is_stereo = true;
  }

  double cur_td;
  Vector3d point, pointRight;
  Vector2d uv, uvRight;
  Vector2d velocity, velocityRight;
  bool is_stereo;
};

class FeaturePerId {
 public:
  FeaturePerId(int _feature_id, int _start_frame)
      : feature_id(_feature_id),
        start_frame(_start_frame),
        used_num(0),
        estimated_depth(-1.0),
        solve_flag(0) {}

  int endFrame() const;

  const int feature_id;
  int start_frame;
  vector<FeaturePerFrame> feature_per_frame;
  int used_num;
  double estimated_depth;
  int solve_flag;  // 0 haven't solve yet; 1 solve succ; 2 solve fail;
  
  // Temporal depth stability tracking (for filtering flickering zero-shot depth)
  std::vector<double> depth_history;  // Circular buffer of aligned inverse depths
  double depth_variance = 1.0;        // Variance of depth_history (default high = unstable)
  bool depth_stable = false;          // True if variance < threshold
  
  // Complementary Boosting Score (1.0 = Fully Trusted Depth, 0.0 = Rejected/Unstable)
  // Used to boost Ordinal/Topological constraints when Metric Depth is unreliable.
  double depth_confidence_score = 1.0;
  
  void updateDepthHistory(double aligned_inv_depth, int buffer_size, double variance_thresh) {
    depth_history.push_back(aligned_inv_depth);
    if (static_cast<int>(depth_history.size()) > buffer_size) {
      depth_history.erase(depth_history.begin());
    }
    // Compute variance if we have enough samples
    if (depth_history.size() >= 3) {
      double mean = 0.0;
      for (double d : depth_history) mean += d;
      mean /= depth_history.size();
      double var = 0.0;
      for (double d : depth_history) var += (d - mean) * (d - mean);
      depth_variance = var / depth_history.size();
      depth_stable = (depth_variance < variance_thresh);
    } else {
      depth_variance = 1.0;
      depth_stable = false;
    }
  }

  // =========================================================================
  // Multi-view depth fusion (Approach 2)
  // Track depth observations from multiple viewpoints and compute Bayesian fusion
  // =========================================================================
  struct DepthObservation {
    double inv_depth;           // Aligned inverse depth observation
    int frame_idx;              // Frame index when observed
    Eigen::Vector3d cam_pos;    // Camera position in world frame when observing
    double uncertainty;         // Uncertainty of this observation (default: 1.0)
  };
  std::vector<DepthObservation> mv_depth_observations;
  double fused_inv_depth = -1.0;     // Bayesian fused inverse depth
  double fused_inv_depth_var = 1.0;  // Fused variance (uncertainty)
  bool has_fused_depth = false;      // True if fusion was computed with enough views
  
  // Add a new depth observation from a viewpoint
  void addDepthObservation(double inv_d, int frame_idx, const Eigen::Vector3d& cam_pos, double uncertainty = 1.0) {
    // Avoid duplicate observations from same frame
    for (const auto& obs : mv_depth_observations) {
      if (obs.frame_idx == frame_idx) return;
    }
    mv_depth_observations.push_back({inv_d, frame_idx, cam_pos, uncertainty});
    
    // Keep only recent observations (sliding window)
    while (mv_depth_observations.size() > 10) {
      mv_depth_observations.erase(mv_depth_observations.begin());
    }
  }
  
  // Compute Bayesian-fused depth from multiple observations
  // Uses inverse-variance weighting: d_fused = Σ(d_i / σ_i²) / Σ(1 / σ_i²)
  void computeFusedDepth(int min_views = 3) {
    if (static_cast<int>(mv_depth_observations.size()) < min_views) {
      has_fused_depth = false;
      return;
    }
    
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    
    for (const auto& obs : mv_depth_observations) {
      double weight = 1.0 / (obs.uncertainty * obs.uncertainty + 1e-8);
      weighted_sum += obs.inv_depth * weight;
      weight_sum += weight;
    }
    
    if (weight_sum > 1e-8) {
      fused_inv_depth = weighted_sum / weight_sum;
      fused_inv_depth_var = 1.0 / weight_sum;  // Fused variance
      has_fused_depth = true;
    } else {
      has_fused_depth = false;
    }
  }
  
  // Clear multi-view data (called on feature removal)
  void clearMVDepth() {
    mv_depth_observations.clear();
    fused_inv_depth = -1.0;
    fused_inv_depth_var = 1.0;
    has_fused_depth = false;
  }
};

class FeatureManager {
 public:
  FeatureManager(Parameters &params);

  void setRic(Matrix3d _ric[]);
  void clearState();
  int getFeatureCount();
  bool addFeatureCheckParallax(
      int frame_count,
      const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
      double td);
  vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l,
                                                    int frame_count_r);
  // void updateDepth(const VectorXd &x);
  void setDepth(const VectorXd &x);
  void removeFailures();
  void clearDepth();
  VectorXd getDepthVector();
  void triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[],
                   Matrix3d ric[]);
  static void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0,
                               Eigen::Matrix<double, 3, 4> &Pose1,
                               Eigen::Vector2d &point0, Eigen::Vector2d &point1,
                               Eigen::Vector3d &point_3d);
  void initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[],
                          Vector3d tic[], Matrix3d ric[]);
  static bool solvePoseByPnP(Eigen::Matrix3d &R_initial,
                             Eigen::Vector3d &P_initial,
                             vector<cv::Point2f> &pts2D,
                             vector<cv::Point3f> &pts3D);
  void removeBackShiftDepth(const Eigen::Matrix3d &marg_R,
                            const Eigen::Vector3d &marg_P,
                            Eigen::Matrix3d new_R,
                            const Eigen::Vector3d &new_P);
  void removeBack();
  void removeFront(int frame_count);
  void removeOutlier(set<int> &outlierIndex);

  static void logFeature(
      const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
      const string &path);
  static void logOutlier(const set<int> &outlierIndex, const string &path);

  list<FeaturePerId> feature;
  int last_track_num;
  double last_average_parallax;
  int new_feature_num;
  int long_track_num;

 private:
  Parameters &params;
  static double compensatedParallax2(const FeaturePerId &it_per_id,
                                     int frame_count);
  const Matrix3d *Rs;
  Matrix3d ric[2];
};

}  // namespace vins::estimator

#endif
