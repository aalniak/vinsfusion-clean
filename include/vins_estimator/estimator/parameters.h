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

#include <ros/ros.h>
#include <vins_estimator/utility/utility.h>

#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace vins::estimator {

#define WINDOW_SIZE 10
#define NUM_OF_F 4096

enum LossType {
  LOSS_L2 = 0,
  LOSS_HUBER = 1,
  LOSS_TUKEY = 2,
  LOSS_CAUCHY = 3,
};

struct Parameters {
  double focal_length;

  double init_depth;
  double min_parallax;
  double acc_n, acc_w;
  double gyr_n, gyr_w;
  std::string depth_engine_path;
  std::string video_depth_engine_path;
  int gating; // 0: Always add prior with weight=1.0, 1: Use variance gating
  std::vector<Eigen::Matrix3d> ric;
  std::vector<Eigen::Vector3d> tic;

  Eigen::Vector3d g{0.0, 0.0, 9.8};

  double bias_acc_threshold;
  double bias_gyr_threshold;
  double solver_time;
  int num_iterations;
  int estimate_extrinsic;
  int estimate_td;
  int rolling_shutter;
  std::string ex_calib_result_path;
  std::string vins_result_path;
  std::string output_folder;
  std::string imu_topic;
  int row, col;
  double td;
  int num_of_cam;
  int stereo;
  int use_imu;
  int multiple_thread;
  
  std::string image0_topic, image1_topic;
  std::string fisheye_mask;
  std::vector<std::string> cam_names;
  int max_cnt;
  int min_dist;
  double f_threshold;
  int show_track;
  int flow_back;
  int min_features;
  std::string pose_graph_save_path;
  int save_image;
  int load_previous_pose_graph;

  double terminate_t;
  std::string depth_folder;
  int feature_debug;
  std::string feature_debug_path;
  bool use_depth;
  int use_gt;
  int use_cuda_in_optimization;
  int use_cuda_in_tracking;
  int rgd;
  int metric_depth_vis = 1;  // 0: inverse depth visualization, 1: metric depth (log-scaled)
  float fx, fy, cx, cy;
  LossType loss_type;
  double loss_parameter;
  LossType loss_type_initial;
  double loss_parameter_initial;

  bool tracking_outlier_rejection;
  bool tracking_prediction;
  
  std::string tapnext_onnx_path;
  std::string tapnext_engine_path;
  bool tapnext_enable;
  int tapnext_max_track;
  float tapnext_reset_boundary_ratio_x;
  float tapnext_reset_boundary_ratio_y;
  float tapnext_reset_min_percent;
  int tapnext_reset_min_count;
  int tapnext_reset_max_frames;

  bool stereo_init;
  int stereo_init_lag;
  
  // Pre-optimization outlier filtering
  int preopt_outlier_filter;           // Enable/disable pre-optimization outlier rejection
  double preopt_edge_threshold;         // Skip features near image edge (normalized coords, e.g. 0.85)
  double preopt_reproj_error_threshold; // Skip features with reproj error above this (normalized coords)
  
  // Depth factor weighting
  int use_mahalanobis_weight;           // 0: constant weight, 1: Mahalanobis distance-based adaptive weight
  int residual_log;                     // 0: linear residual (inv_d - inv_d_prior), 1: log residual (log(inv_d) - log(inv_d_prior))

  // Temporal stability filter for depth priors
  int temporal_stable;                    // 0: disabled, 1: enabled - only trust temporally stable depth
  int temporal_stable_buffer_size;        // Number of frames to track depth history (default: 5)
  double temporal_stable_variance_thresh; // Max variance (in inv-depth) to trust depth (default: 0.01)

  // Ordinal depth constraints (relative ordering)
  int ordinal_depth;                      // 0: disabled, 1: enabled - enforce relative depth ordering
  double ordinal_depth_margin;            // Min inv-depth difference to enforce ordering (default: 0.01)
  double ordinal_depth_weight;            // Weight for ordinal constraint factors (default: 1.0)
  int ordinal_depth_max_pairs;            // Max number of ordinal pairs per frame (default: 50)
  double ordinal_depth_max_dist;          // Max spatial distance (pixels) between paired features
  int ordinal_grid_enable;                // [LOKI] Enable Grid Filtering
  double ordinal_depth_max_metric;        // [LOKI] Max metric depth to use (e.g. 50m)
  int save_ordinal_debug;                 // [LOKI] Save debug images to disk
  int ordinal_temporal_consistency;       // [LOKI] 0: disabled, 1: enabled - check depth history
  int ordinal_consistency_min_frames;     // [LOKI] Min frames to verify (default 1)

  // Multi-view depth fusion (Approach 2)
  int mv_depth_fusion;                      // 0: disabled, 1: enabled - fuse depth across multiple viewpoints
  int mv_depth_min_views;                   // Min viewpoints before trusting fused depth (default: 3)
  double mv_depth_fusion_weight;            // Weight for fused depth prior factor (default: 1.0)

  // Photometric regularization (Approach 3)
  int photometric_reg;                      // 0: disabled, 1: enabled - image warping loss
  double photometric_weight;                // Weight for photometric loss (default: 0.1)
  int photometric_keyframe_gap;             // Min frame gap between keyframe pairs (default: 2)
  double photometric_ssim_weight;           // SSIM portion weight (default: 0.85)
  double photometric_l1_weight;             // L1 portion weight (default: 0.15)
  
  int video_mode;                           // 0: default, 1: video/stateful


  int diagnostics;

  void read_from_file(const std::string &config_file);
};

enum SIZE_PARAMETERIZATION {
  SIZE_POSE = 7,
  SIZE_SPEEDBIAS = 9,
  SIZE_FEATURE = 1
};

enum StateOrder { O_P = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12 };

enum NoiseOrder { O_AN = 0, O_GN = 3, O_AW = 6, O_GW = 9 };

}  // namespace vins::estimator
