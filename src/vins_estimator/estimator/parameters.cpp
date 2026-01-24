/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vins_estimator/estimator/parameters.h>

#include <fstream>

namespace vins::estimator {

void Parameters::read_from_file(const std::string &config_file) {
  FILE *fh = fopen(config_file.c_str(), "r");
  if (fh == nullptr) {
    ROS_WARN("config_file doesn't exist; wrong config_file path");
    ROS_BREAK();
    return;
  }
  fclose(fh);

  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings" << std::endl;
  }

  fsSettings["image0_topic"] >> image0_topic;
  fsSettings["image1_topic"] >> image1_topic;
  max_cnt = fsSettings["max_cnt"];
  min_dist = fsSettings["min_dist"];
  f_threshold = fsSettings["F_threshold"];
  show_track = fsSettings["show_track"];
  flow_back = fsSettings["flow_back"];

  multiple_thread = fsSettings["multiple_thread"];

  use_imu = fsSettings["imu"];
  printf("USE_IMU: %d\n", use_imu);
  if (use_imu) {
    fsSettings["imu_topic"] >> imu_topic;
    printf("IMU_TOPIC: %s\n", imu_topic.c_str());
    acc_n = fsSettings["acc_n"];
    acc_w = fsSettings["acc_w"];
    gyr_n = fsSettings["gyr_n"];
    gyr_w = fsSettings["gyr_w"];
    g.z() = fsSettings["g_norm"];
  }

  // check if focal_length is set
  if (fsSettings["focal_length"].empty()) {
    std::cerr << "ERROR: focal_length not set in config file" << std::endl;
    focal_length = 460.0;
  } else {
    fsSettings["focal_length"] >> focal_length;
    std::cout << "focal length: " << focal_length << std::endl;
  }

  solver_time = fsSettings["max_solver_time"];
  num_iterations = fsSettings["max_num_iterations"];
  min_parallax = fsSettings["keyframe_parallax"];
  min_parallax /= focal_length;
  fsSettings["min_features"] >> min_features;
  fsSettings["output_path"] >> output_folder;
  vins_result_path = output_folder + "/vio.csv";
  feature_debug_path = output_folder + "/feature_debug.csv";

  std::cout << "result path " << vins_result_path << std::endl;
  std::ofstream fout(vins_result_path, std::ios::out);
  fout.close();

  std::cout << "feature debug path " << feature_debug_path << std::endl;
  std::ofstream ffeature_debug(feature_debug_path, std::ios::out);
  ffeature_debug.close();

  // check for feature_debug bool, if not exist, set to false
  if (!fsSettings["feature_debug"].empty()) {
    fsSettings["feature_debug"] >> feature_debug;
    if (feature_debug) {
      std::cout << "feature debug enabled" << std::endl;
    } else {
      std::cout << "feature debug disabled" << std::endl;
    }
  } else {
    feature_debug = 0;
    std::cout << "feature debug not set, default to disabled" << std::endl;
  }

  estimate_extrinsic = fsSettings["estimate_extrinsic"];
  if (estimate_extrinsic == 2) {
    ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
    ric.emplace_back(Eigen::Matrix3d::Identity());
    tic.emplace_back(Eigen::Vector3d::Zero());
    ex_calib_result_path = output_folder + "/extrinsic_parameter.csv";
  } else {
    if (estimate_extrinsic == 1) {
      ROS_WARN(" Optimize extrinsic param around initial guess!");
      ex_calib_result_path = output_folder + "/extrinsic_parameter.csv";
    }
    if (estimate_extrinsic == 0) ROS_WARN(" fix extrinsic param ");

    cv::Mat cv_T;
    fsSettings["body_T_cam0"] >> cv_T;
    Eigen::Matrix4d T;
    cv::cv2eigen(cv_T, T);
    ric.emplace_back(T.block<3, 3>(0, 0));
    tic.emplace_back(T.block<3, 1>(0, 3));
  }

  num_of_cam = fsSettings["num_of_cam"];
  printf("camera number %d\n", num_of_cam);

  if (num_of_cam != 1 && num_of_cam != 2) {
    printf("num_of_cam should be 1 or 2\n");
    assert(0);
  }

  int pn = config_file.find_last_of('/');
  std::string configPath = config_file.substr(0, pn);

  std::string cam0Calib;
  fsSettings["cam0_calib"] >> cam0Calib;
  std::string cam0Path = configPath + "/" + cam0Calib;
  cam_names.push_back(cam0Path);

  if (num_of_cam == 2) {
    stereo = 1;
    std::string cam1Calib;
    fsSettings["cam1_calib"] >> cam1Calib;
    std::string cam1Path = configPath + "/" + cam1Calib;
    // printf("%s cam1 path\n", cam1Path.c_str() );
    cam_names.push_back(cam1Path);

    cv::Mat cv_T;
    fsSettings["body_T_cam1"] >> cv_T;
    Eigen::Matrix4d T;
    cv::cv2eigen(cv_T, T);
    ric.emplace_back(T.block<3, 3>(0, 0));
    tic.emplace_back(T.block<3, 1>(0, 3));
  }

  init_depth = 5.0;
  bias_acc_threshold = 0.1;
  bias_gyr_threshold = 0.1;

  td = fsSettings["td"];
  estimate_td = fsSettings["estimate_td"];
  if (estimate_td)
    ROS_INFO_STREAM(
        "Unsynchronized sensors, online estimate time offset, initial td: "
        << td);
  else
    ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << td);

  row = fsSettings["image_height"];
  col = fsSettings["image_width"];
  ROS_INFO("ROW: %d COL: %d ", row, col);

  if (!use_imu) {
    estimate_extrinsic = 0;
    estimate_td = 0;
    printf("no imu, fix extrinsic param; no time offset calibration\n");
  }

  fsSettings["pose_graph_save_path"] >> pose_graph_save_path;

  fsSettings["use_depth"] >> use_depth;
  fsSettings["use_cuda_in_optimization"] >> use_cuda_in_optimization;
  fsSettings["use_cuda_in_tracking"] >> use_cuda_in_tracking;
  fsSettings["rgd"] >> rgd;
  fsSettings["metric_depth_vis"] >> metric_depth_vis;
  if (metric_depth_vis != 0 && metric_depth_vis != 1) metric_depth_vis = 1;  // Default to metric
  fsSettings["fx"] >> fx;
  fsSettings["fy"] >> fy;
  fsSettings["cx"] >> cx;
  fsSettings["cy"] >> cy;
  fsSettings["depth_folder"] >> depth_folder;
  fsSettings["use_gt"] >> use_gt;
  fsSettings["depth_engine_path"] >> depth_engine_path;
  save_image = fsSettings["save_image"];
  load_previous_pose_graph = fsSettings["load_previous_pose_graph"];

  char *env_terminate_t_str = getenv("VINS_TERMINATE_TIME");
  if (env_terminate_t_str != nullptr) {
    terminate_t = atof(env_terminate_t_str);
  } else {
    // default value, no termination time set
    terminate_t = -1.0;
  }

  // check if loss_type is set
  if (fsSettings["loss_type"].empty()) {
    std::cerr
        << "ERROR: loss_type not set in config file, defaulting to LOSS_L2"
        << std::endl;
    loss_type = LOSS_HUBER;
  } else {
    int loss_type_int;
    fsSettings["loss_type"] >> loss_type_int;
    loss_type = static_cast<LossType>(loss_type_int);
  }

  // check if loss_parameter is set
  if (fsSettings["loss_parameter"].empty()) {
    std::cerr
        << "ERROR: loss_parameter not set in config file, defaulting to 1.0"
        << std::endl;
    loss_parameter = 1.0;
  } else {
    fsSettings["loss_parameter"] >> loss_parameter;
  }

  // check if loss_type_initial is set
  if (fsSettings["loss_type_initial"].empty()) {
    std::cerr << "ERROR: loss_type_initial not set in config file, defaulting "
                 "to LOSS_L2"
              << std::endl;
    loss_type_initial = LOSS_L2;
  } else {
    int loss_type_initial_int;
    fsSettings["loss_type_initial"] >> loss_type_initial_int;
    loss_type_initial = static_cast<LossType>(loss_type_initial_int);
  }

  // check if loss_parameter_initial is set
  if (fsSettings["loss_parameter_initial"].empty()) {
    std::cerr << "ERROR: loss_parameter_initial not set in config file, "
                 "defaulting to 1.0"
              << std::endl;
    loss_parameter_initial = 1.0;
  } else {
    fsSettings["loss_parameter_initial"] >> loss_parameter_initial;
  }

  // check tracking_outlier_rejection
  if (fsSettings["tracking_outlier_rejection"].empty()) {
    std::cerr << "ERROR: tracking_outlier_rejection not set in config file, "
                 "defaulting to false"
              << std::endl;
    tracking_outlier_rejection = false;
  } else {
    fsSettings["tracking_outlier_rejection"] >> tracking_outlier_rejection;
  }

  if (fsSettings["tracking_prediction"].empty()) {
    std::cerr << "ERROR: tracking_prediction not set in config file, "
                 "defaulting to false"
              << std::endl;
    tracking_prediction = false;
  } else {
    fsSettings["tracking_prediction"] >> tracking_prediction;
  }

  if (fsSettings["stereo_init"].empty()) {
    std::cerr << "ERROR: stereo_init not set in config file, "
                 "defaulting to false"
              << std::endl;
    stereo_init = false;
  } else {
    fsSettings["stereo_init"] >> stereo_init;
  }

  if (fsSettings["stereo_init_lag"].empty()) {
    std::cerr << "ERROR: stereo_init_lag not set in config file, "
                 "defaulting to 0"
              << std::endl;
    stereo_init_lag = 0;
  } else {
    fsSettings["stereo_init_lag"] >> stereo_init_lag;
  }

  // Pre-optimization outlier filtering parameters
  if (fsSettings["preopt_outlier_filter"].empty()) {
    preopt_outlier_filter = 0;  // Default: disabled
  } else {
    fsSettings["preopt_outlier_filter"] >> preopt_outlier_filter;
  }
  
  if (fsSettings["preopt_edge_threshold"].empty()) {
    preopt_edge_threshold = 0.85;  // Default: skip features with |x|,|y| > 0.85
  } else {
    fsSettings["preopt_edge_threshold"] >> preopt_edge_threshold;
  }
  
  if (fsSettings["preopt_reproj_error_threshold"].empty()) {
    preopt_reproj_error_threshold = 1.0;  // Default: skip features with reproj error > 1.0
  } else {
    fsSettings["preopt_reproj_error_threshold"] >> preopt_reproj_error_threshold;
  }
  
  if (preopt_outlier_filter) {
    ROS_INFO("\033[1;32m[PRE-OPT] Outlier filter ENABLED: edge_thresh=%.2f, reproj_thresh=%.2f\033[0m",
             preopt_edge_threshold, preopt_reproj_error_threshold);
  } else {
    ROS_INFO("\033[1;33m[PRE-OPT] Outlier filter DISABLED\033[0m");
  }

  // Depth factor Mahalanobis weighting
  if (fsSettings["use_mahalanobis_weight"].empty()) {
    use_mahalanobis_weight = 0;  // Default: disabled (constant weight)
  } else {
    fsSettings["use_mahalanobis_weight"] >> use_mahalanobis_weight;
  }
  
  if (use_mahalanobis_weight) {
    ROS_INFO("\033[1;32m[DEPTH] Mahalanobis distance-based adaptive weighting ENABLED\033[0m");
  } else {
    ROS_INFO("\033[1;33m[DEPTH] Using constant depth factor weight\033[0m");
  }

  // Depth factor residual domain
  if (fsSettings["residual_log"].empty()) {
    residual_log = 0;  // Default: linear residual
  } else {
    fsSettings["residual_log"] >> residual_log;
  }
  
  if (residual_log) {
    ROS_INFO("\033[1;32m[DEPTH] Using LOG domain residual: log(inv_d_vio) - log(inv_d_prior)\033[0m");
  } else {
    ROS_INFO("\033[1;33m[DEPTH] Using LINEAR residual: inv_d_vio - inv_d_prior\033[0m");
  }

  // Temporal stability filter for depth priors
  if (fsSettings["temporal_stable"].empty()) {
    temporal_stable = 0;  // Default: disabled
  } else {
    fsSettings["temporal_stable"] >> temporal_stable;
  }
  
  if (fsSettings["temporal_stable_buffer_size"].empty()) {
    temporal_stable_buffer_size = 5;  // Default: 5 frames
  } else {
    fsSettings["temporal_stable_buffer_size"] >> temporal_stable_buffer_size;
  }
  
  if (fsSettings["temporal_stable_variance_thresh"].empty()) {
    temporal_stable_variance_thresh = 0.01;  // Default: 0.01 inv-depth variance
  } else {
    fsSettings["temporal_stable_variance_thresh"] >> temporal_stable_variance_thresh;
  }
  
  if (temporal_stable) {
    ROS_INFO("\033[1;32m[DEPTH] Temporal stability filter ENABLED: buffer=%d, var_thresh=%.4f\033[0m",
             temporal_stable_buffer_size, temporal_stable_variance_thresh);
  } else {
    ROS_INFO("\033[1;33m[DEPTH] Temporal stability filter DISABLED\033[0m");
  }

  // Ordinal depth constraints
  if (fsSettings["ordinal_depth"].empty()) {
    ordinal_depth = 0;  // Default: disabled
  } else {
    fsSettings["ordinal_depth"] >> ordinal_depth;
  }
  
  if (fsSettings["ordinal_depth_margin"].empty()) {
    ordinal_depth_margin = 0.01;  // Default: 0.01 inv-depth margin
  } else {
    fsSettings["ordinal_depth_margin"] >> ordinal_depth_margin;
  }
  
  if (fsSettings["ordinal_depth_weight"].empty()) {
    ordinal_depth_weight = 1.0;  // Default: 1.0
  } else {
    fsSettings["ordinal_depth_weight"] >> ordinal_depth_weight;
  }
  
  if (fsSettings["ordinal_depth_max_pairs"].empty()) {
    ordinal_depth_max_pairs = 50;  // Default: 50 pairs per frame
  } else {
    fsSettings["ordinal_depth_max_pairs"] >> ordinal_depth_max_pairs;
  }
  
  if (ordinal_depth) {
    ROS_INFO("\033[1;32m[DEPTH] Ordinal depth constraints ENABLED: margin=%.4f, weight=%.2f, max_pairs=%d\033[0m",
             ordinal_depth_margin, ordinal_depth_weight, ordinal_depth_max_pairs);
  } else {
    ROS_INFO("\033[1;33m[DEPTH] Ordinal depth constraints DISABLED\033[0m");
  }

  if (fsSettings["tapnext_onnx_path"].empty()) {
    std::cerr << "ERROR: tapnext_onnx_path not set in config file, "
                 "defaulting to empty string"
              << std::endl;
    tapnext_onnx_path = "";
  } else {
    fsSettings["tapnext_onnx_path"] >> tapnext_onnx_path;
  }

  if (fsSettings["tapnext_engine_path"].empty()) {
    std::cerr << "ERROR: tapnext_engine_path not set in config file, "
                 "defaulting to empty string"
              << std::endl;
    tapnext_engine_path = "";
  } else {
    fsSettings["tapnext_engine_path"] >> tapnext_engine_path;
  }

  if (fsSettings["tapnext_enable"].empty()) {
    std::cerr << "ERROR: tapnext_enable not set in config file, "
                 "defaulting to false"
              << std::endl;
    tapnext_enable = false;
  } else {
    fsSettings["tapnext_enable"] >> tapnext_enable;
  }

  if (fsSettings["tapnext_max_track"].empty()) {
    std::cerr << "ERROR: tapnext_max_track not set in config file, "
                 "defaulting to 256"
              << std::endl;
    tapnext_max_track = 256;
  } else {
    fsSettings["tapnext_max_track"] >> tapnext_max_track;
  }

  if (fsSettings["tapnext_reset_boundary_ratio_x"].empty()) {
    std::cerr << "ERROR: tapnext_reset_boundary_ratio_x not set in config file, "
                 "defaulting to 0.3"
              << std::endl;
    tapnext_reset_boundary_ratio_x = 0.3F;
  } else {
    fsSettings["tapnext_reset_boundary_ratio_x"] >> tapnext_reset_boundary_ratio_x;
  }
  
  if (fsSettings["tapnext_reset_boundary_ratio_y"].empty()) {
    std::cerr << "ERROR: tapnext_reset_boundary_ratio_y not set in config file, "
                 "defaulting to 1.0"
              << std::endl;
    tapnext_reset_boundary_ratio_y = 1.0F;
  } else {
    fsSettings["tapnext_reset_boundary_ratio_y"] >> tapnext_reset_boundary_ratio_y;
  }

  if (fsSettings["tapnext_reset_min_percent"].empty()) {
    std::cerr << "ERROR: tapnext_reset_min_percent not set in config file, "
                 "defaulting to 0.1"
              << std::endl;
    tapnext_reset_min_percent = 0.1F;
  } else {
    fsSettings["tapnext_reset_min_percent"] >> tapnext_reset_min_percent;
  }

  if (fsSettings["tapnext_reset_min_count"].empty()) {
    std::cerr << "ERROR: tapnext_reset_min_count not set in config file, "
                 "defaulting to 10"
              << std::endl;
    tapnext_reset_min_count = 10;
  } else {
    fsSettings["tapnext_reset_min_count"] >> tapnext_reset_min_count;
  }

    if (fsSettings["tapnext_reset_max_frames"].empty()) {
        std::cerr << "ERROR: tapnext_reset_max_frames not set in config file, "
                        "defaulting to 100"
                << std::endl;
        tapnext_reset_max_frames = 100;
    } else {
        fsSettings["tapnext_reset_max_frames"] >> tapnext_reset_max_frames;
    }

  fsSettings.release();

  std::cout << "loss type: " << loss_type
            << ", loss parameter: " << loss_parameter << std::endl;
  std::cout << "loss type initial: " << loss_type_initial
            << ", loss parameter initial: " << loss_parameter_initial
            << std::endl;
}

}  // namespace vins::estimator
