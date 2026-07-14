/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <vins_estimator/featureTracker/feature_tracker_klt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
namespace vins::estimator {
bool ready = false;
bool FeatureTrackerKLT::inBorder(const cv::Point2f &pt) const {
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE &&
         BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(const cv::Point2f &pt1, const cv::Point2f &pt2) {
  // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
  double dx = pt1.x - pt2.x;
  double dy = pt1.y - pt2.y;
  return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < static_cast<int>(v.size()); i++)
    if (status[i]) v[j++] = v[i];
  v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < static_cast<int>(v.size()); i++)
    if (status[i]) v[j++] = v[i];
  v.resize(j);
}

FeatureTrackerKLT::FeatureTrackerKLT(Parameters &params)
    : params(params), stereo_cam_(false), has_prediction_(false)
{
    if (params.use_cuda_in_tracking){
    gpu_lk_tracker = cv::cuda::SparsePyrLKOpticalFlow::create(
        cv::Size(21, 21), 3, 30, true);
        
    gpu_detector = cv::cuda::createGoodFeaturesToTrackDetector(
        CV_8UC1, params.max_cnt, 0.01, params.min_dist);
    mem_cur_img = cv::cuda::HostMem(params.row, params.col, CV_8UC1, cv::cuda::HostMem::SHARED);
    mem_prev_pts = cv::cuda::HostMem(1, params.max_cnt, CV_32FC2, cv::cuda::HostMem::SHARED);
    mem_cur_pts = cv::cuda::HostMem(1, params.max_cnt, CV_32FC2, cv::cuda::HostMem::SHARED);
    mem_status = cv::cuda::HostMem(1, params.max_cnt, CV_8UC1, cv::cuda::HostMem::SHARED);
    mem_reverse_pts = cv::cuda::HostMem(1, params.max_cnt, CV_32FC2, cv::cuda::HostMem::SHARED);
    mem_reverse_status = cv::cuda::HostMem(1, params.max_cnt, CV_8UC1, cv::cuda::HostMem::SHARED);
    cpu_cur_img_view = mem_cur_img.createMatHeader();
    gpu_cur_img_view = mem_cur_img.createGpuMatHeader();
    //cv::cuda::HostMem shared_img_mem(HEIGHT, WIDTH, CV_8UC1, cv::cuda::HostMem::SHARED);
    }
}

void FeatureTrackerKLT::setMask() {
  mask_ = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

  // prefer to keep features that are tracked for long time
  vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

  for (unsigned i = 0; i < cur_pts_.size(); i++)
    cnt_pts_id.emplace_back(track_cnt_[i], make_pair(cur_pts_[i], ids_[i]));

  sort(cnt_pts_id.begin(), cnt_pts_id.end(),
       [](const pair<int, pair<cv::Point2f, int>> &a,
          const pair<int, pair<cv::Point2f, int>> &b) {
         return a.first > b.first;
       });

  cur_pts_.clear();
  ids_.clear();
  track_cnt_.clear();

  for (auto &it : cnt_pts_id) {
    if (mask_.at<uchar>(it.second.first) == 255) {
      cur_pts_.push_back(it.second.first);
      ids_.push_back(it.second.second);
      track_cnt_.push_back(it.first);
      cv::circle(mask_, it.second.first, params.min_dist, 0, -1);
    }
  }
}

double FeatureTrackerKLT::distance(const cv::Point2f &pt1,
                                const cv::Point2f &pt2) {
  // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
  double dx = pt1.x - pt2.x;
  double dy = pt1.y - pt2.y;
  return sqrt(dx * dx + dy * dy);
}


cv::Mat createDualChannelInput(const cv::Mat& gray, const cv::Mat& depth) {
    cv::Mat depth_8u;

    // 1. SAFETY CHECK: Is depth empty or wrong size?
    if (depth.empty() || depth.size() != gray.size()) {
        // Fallback: Create a black depth image (all zeros)
        // This allows KLT to run purely on the gray channel (channel 0) 
        // without crashing during startup.
        depth_8u = cv::Mat::zeros(gray.size(), CV_8UC1);
    } 
    else {
        // 2. Normalize and Scale Real Depth
        // Assuming raw_depth is CV_32F or CV_16U, we need CV_8U for KLT
        depth_8u = depth_8u * 1.0;
        }

    // 4. Merge into 2-Channel Mat
    std::vector<cv::Mat> channels;
    channels.push_back(gray);     // Channel 0: Texture
    channels.push_back(depth_8u); // Channel 1: Depth (or zeros)

    cv::Mat merged;
    cv::merge(channels, merged);
    
    return merged;
}


map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>>
FeatureTrackerKLT::trackImage(double _cur_time, const cv::Mat &_img,
                           const cv::Mat &_img1) {
  TicToc t_r;
  cur_time_ = _cur_time;
  cur_img_ = _img;
  row = cur_img_.rows;
  col = cur_img_.cols;
  const cv::Mat &rightImg = _img1;
  /*
  {
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
      clahe->apply(cur_img, cur_img);
      if(!rightImg.empty())
          clahe->apply(rightImg, rightImg);
  }
  */
  cur_pts_.clear();

  // Extract XFeat for this frame (+ optional LighterGlue-guided KLT init).
  extractAndGuide();

  if (prev_pts_.size() > 0) {
    TicToc t_o;
    vector<uchar> status;
    // Semi-dense: track each feature by matching the dense XFeat descriptor map
    // locally around its predicted position (replaces KLT optical flow). CPU path
    // only (the deployed path; use_cuda_in_tracking:0).
    if (params.xfeat_semidense && xfeat_ && xfeat_->hasDense()) {
      trackDense(status);
    } else {
    vector<float> err;
    if (has_prediction_) {
      cur_pts_ = predict_pts_;
      cv::calcOpticalFlowPyrLK(
          prev_img_, cur_img_, prev_pts_, cur_pts_, status, err,
          cv::Size(21, 21), 3,
          cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                           0.01),
          cv::OPTFLOW_USE_INITIAL_FLOW);

      int succ_num = 0;
      for (unsigned char statu : status) {
        if (statu) succ_num++;
      }
      if (succ_num < 10)
        cv::calcOpticalFlowPyrLK(prev_img_, cur_img_, prev_pts_, cur_pts_,
                                 status, err, cv::Size(21, 21), 3);
    } else {
      cv::calcOpticalFlowPyrLK(prev_img_, cur_img_, prev_pts_, cur_pts_, status,
                               err, cv::Size(21, 21), 3);
      std::cout << "time it takes for optical flow: " << t_o.toc() << " ms"
                << std::endl;
      }
    // reverse check
    if (params.flow_back) {
      vector<uchar> reverse_status;
      vector<cv::Point2f> reverse_pts = prev_pts_;
      cv::calcOpticalFlowPyrLK(
          cur_img_, prev_img_, cur_pts_, reverse_pts, reverse_status, err,
          cv::Size(21, 21), 1,
          cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                           0.01),
          cv::OPTFLOW_USE_INITIAL_FLOW);
      // cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts,
      // reverse_status, err, cv::Size(21, 21), 3);
      for (size_t i = 0; i < status.size(); i++) {
        if (status[i] && reverse_status[i] &&
            distance(prev_pts_[i], reverse_pts[i]) <= 0.5) {
          status[i] = 1;
        } else
          status[i] = 0;
      }
      std::cout << "time it takes for reverse optical flow: " << t_o.toc() << " ms" << std::endl;
    }

    // Recover KLT-lost tracks via LighterGlue, then drop appearance-drifted tracks,
    // before culling (both no-ops unless their params are enabled).
    recoverLostTracks(status);
    recoverFromKeyframe(status);
    cleanDriftedTracks(status);
    }  // end KLT vs semi-dense branch

    for (int i = 0; i < static_cast<int>(cur_pts_.size()); i++)
      if (status[i] && !inBorder(cur_pts_[i])) status[i] = 0;
    reduceVector(prev_pts_, status);
    reduceVector(cur_pts_, status);
    reduceVector(ids_, status);
    reduceVector(track_cnt_, status);
    ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    // printf("track cnt %d\n", (int)ids.size());
  }

  for (auto &n : track_cnt_) n++;

  // Drop persistent rigid-motion violators (moving objects); no-op unless xfeat_dyn_mask.
  maskDynamicTracks();

  // if (true)
  {
    // rejectWithF();
    ROS_DEBUG("set mask begins");
    TicToc t_m;
    setMask();
    ROS_DEBUG("set mask costs %fms", t_m.toc());

    ROS_DEBUG("detect feature begins");
    TicToc t_t;
    int n_max_cnt = params.max_cnt - static_cast<int>(cur_pts_.size());
    if (n_max_cnt > 0) {
      if (mask_.empty()) cout << "mask is empty " << endl;
      if (mask_.type() != CV_8UC1) cout << "mask type wrong " << endl;

      if (params.xfeat_enable && xfeat_) {
        // Hybrid: seed new features from XFeat (same as the CUDA path) so the
        // hybrid works even with use_cuda_in_tracking:0.
        detectNewFeatures(n_max_cnt);
      } else {
        vector<cv::Point2f> n_pts;
        n_pts.reserve(n_max_cnt);
        cv::goodFeaturesToTrack(cur_img_, n_pts, n_max_cnt, 0.01,
                                params.min_dist, mask_);
        for (auto &p : n_pts) {
          cur_pts_.push_back(p);
          ids_.push_back(IdCounter::get());
          track_cnt_.push_back(1);
        }
      }
    }

    ROS_INFO("detect feature costs: %f ms", t_t.toc());
    // printf("feature cnt after add %d\n", (int)ids.size());
  }
  // Snapshot a keyframe for wide-baseline recovery (no-op unless xfeat_kf_recover).
  maybeUpdateKeyframe();

  TicToc t_und;
  cur_un_pts_ = undistortedPts(cur_pts_, m_camera_[0]);
  pts_velocity_ =
      ptsVelocity(ids_, cur_un_pts_, cur_un_pts_map_, prev_un_pts_map_);
  std::cout << "undistort pts + velcalc costs: " << t_und.toc() << " ms"
            << std::endl;
  if (!_img1.empty() && stereo_cam_) {
    ids_right_.clear();
    cur_right_pts_.clear();
    cur_un_right_pts_.clear();
    right_pts_velocity_.clear();
    cur_un_right_pts_map_.clear();
    if (!cur_pts_.empty()) {
      // printf("stereo image; track feature on right image\n");
      vector<cv::Point2f> reverseLeftPts;
      vector<uchar> status;
      vector<uchar> statusRightLeft;
      vector<float> err;
      // cur left ---- cur right
      cv::calcOpticalFlowPyrLK(cur_img_, rightImg, cur_pts_, cur_right_pts_,
                               status, err, cv::Size(21, 21), 3);
      // reverse check cur right ---- cur left
      if (params.flow_back) {
        cv::calcOpticalFlowPyrLK(rightImg, cur_img_, cur_right_pts_,
                                 reverseLeftPts, statusRightLeft, err,
                                 cv::Size(21, 21), 3);
        for (size_t i = 0; i < status.size(); i++) {
          if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts_[i]) &&
              distance(cur_pts_[i], reverseLeftPts[i]) <= 0.5)
            status[i] = 1;
          else
            status[i] = 0;
        }
      }

      ids_right_ = ids_;
      reduceVector(cur_right_pts_, status);
      reduceVector(ids_right_, status);
      // only keep left-right pts
      /*
      reduceVector(cur_pts, status);
      reduceVector(ids, status);
      reduceVector(track_cnt, status);
      reduceVector(cur_un_pts, status);
      reduceVector(pts_velocity, status);
      */
      cur_un_right_pts_ = undistortedPts(cur_right_pts_, m_camera_[1]);
      right_pts_velocity_ =
          ptsVelocity(ids_right_, cur_un_right_pts_, cur_un_right_pts_map_,
                      prev_un_right_pts_map_);
    }
    prev_un_right_pts_map_ = cur_un_right_pts_map_;
  }
  if (params.show_track)
    drawTrack(cur_img_, rightImg, ids_, cur_pts_, cur_right_pts_,
              prev_left_pts_map_);

  prev_img_ = cur_img_;
  prev_pts_ = cur_pts_;
  prev_un_pts_ = cur_un_pts_;
  prev_un_pts_map_ = cur_un_pts_map_;
  prev_time_ = cur_time_;
  has_prediction_ = false;

  prev_left_pts_map_.clear();
  for (size_t i = 0; i < cur_pts_.size(); i++)
    prev_left_pts_map_[ids_[i]] = cur_pts_[i];

  map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> featureFrame;
  for (size_t i = 0; i < ids_.size(); i++) {
    int feature_id = ids_[i];
    double x = cur_un_pts_[i].x;
    double y = cur_un_pts_[i].y;
    double z = 1;
    double p_u = cur_pts_[i].x;
    double p_v = cur_pts_[i].y;
    int camera_id = 0;
    double velocity_x = pts_velocity_[i].x;
    double velocity_y = pts_velocity_[i].y;

    Eigen::Matrix<double, 8, 1> xyz_uv_velocity;
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
    featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
  }

  if (!_img1.empty() && stereo_cam_) {
    for (size_t i = 0; i < ids_right_.size(); i++) {
      int feature_id = ids_right_[i];
      double x = cur_un_right_pts_[i].x;
      double y = cur_un_right_pts_[i].y;
      double z = 1;
      double p_u = cur_right_pts_[i].x;
      double p_v = cur_right_pts_[i].y;
      int camera_id = 1;
      double velocity_x = right_pts_velocity_[i].x;
      double velocity_y = right_pts_velocity_[i].y;

      Eigen::Matrix<double, 8, 1> xyz_uv_velocity;
      xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
      featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }
  }

  printf("feature track whole time %f\n", t_r.toc());
  return featureFrame;
}

map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>>
FeatureTrackerKLT::trackImageCUDA(double _cur_time, const cv::Mat &_img,
                           const cv::Mat &_img1) {
    TicToc t_r;
    cur_time_ = _cur_time;
    cur_img_ = _img;
    row = cur_img_.rows;
    col = cur_img_.cols;
    const cv::Mat &rightImg = _img1;
    std::cout << "Tracking image using CUDA!" << std::endl;
    cur_pts_.clear();

    // [GPU] 1. Upload Current Image to GPU
    // We do this ONCE at the start so both tracking and detection can use it
    d_cur_img.upload(cur_img_);

    // Extract XFeat for this frame (+ optional LighterGlue-guided KLT init).
    extractAndGuide();

    if (prev_pts_.size() > 0) {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;

        // [GPU] 2. Upload Previous Points
        // We must reshape vector<Point2f> to a 1xN matrix for CUDA
        cv::Mat prev_pts_mat(1, prev_pts_.size(), CV_32FC2, (void *)&prev_pts_[0]);
        d_prev_pts.upload(prev_pts_mat);

        // [GPU] 3. Handle Prediction (IMU guess)
        if (has_prediction_) {
            // If we have a guess, upload it as the "initial flow" for d_cur_pts
            cv::Mat predict_pts_mat(1, predict_pts_.size(), CV_32FC2, (void *)&predict_pts_[0]);
            d_cur_pts.upload(predict_pts_mat);
            
            // Run LK with USE_INITIAL_FLOW flag
            gpu_lk_tracker->setUseInitialFlow(true);
            gpu_lk_tracker->calc(d_prev_img, d_cur_img, d_prev_pts, d_cur_pts, d_status, d_err);
        } else {
            TicToc t_track;
            // No guess, standard tracking
            gpu_lk_tracker->setUseInitialFlow(false);
            gpu_lk_tracker->calc(d_prev_img, d_cur_img, d_prev_pts, d_cur_pts, d_status, d_err);
            ROS_INFO("GPU Tracking Time: %f ms", t_track.toc());
        }

        // [GPU] 4. Forward-Backward Check (Robustness)
        if (params.flow_back) {
            // Track BACKWARDS: Current -> Previous
            // We use the 'd_cur_pts' we just calculated as the starting point
            gpu_lk_tracker->setUseInitialFlow(false); // Usually no prediction for reverse
            gpu_lk_tracker->calc(d_cur_img, d_prev_img, d_cur_pts, d_reverse_pts, d_reverse_status);

            // [GPU -> CPU] Download everything to do the distance check
            // Logic is easier on CPU than writing a custom CUDA kernel
            vector<cv::Point2f> tmp_cur_pts(d_cur_pts.cols);
            vector<cv::Point2f> tmp_rev_pts(d_reverse_pts.cols);
            vector<uchar> tmp_status(d_status.cols);
            vector<uchar> tmp_rev_status(d_reverse_status.cols);

            cv::Mat tmp_mat;
            
            d_cur_pts.download(tmp_mat);
            tmp_mat.copyTo(cv::Mat(1, d_cur_pts.cols, CV_32FC2, &tmp_cur_pts[0]));

            d_reverse_pts.download(tmp_mat);
            tmp_mat.copyTo(cv::Mat(1, d_reverse_pts.cols, CV_32FC2, &tmp_rev_pts[0]));

            d_status.download(tmp_mat);
            tmp_mat.copyTo(cv::Mat(1, d_status.cols, CV_8UC1, &tmp_status[0]));
            
            d_reverse_status.download(tmp_mat);
            tmp_mat.copyTo(cv::Mat(1, d_reverse_status.cols, CV_8UC1, &tmp_rev_status[0]));

            // Verify Tracks
            cur_pts_ = tmp_cur_pts; // Update class member
            status = tmp_status;    // Update local status
            
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i] && tmp_rev_status[i] &&
                    distance(prev_pts_[i], tmp_rev_pts[i]) <= 0.5) {
                    status[i] = 1;
                } else {
                    status[i] = 0;
                }
            }
            
        }
        else {
             // If no flow_back, just download the forward results
             vector<cv::Point2f> tmp_cur_pts(d_cur_pts.cols);
             vector<uchar> tmp_status(d_status.cols);
             
             cv::Mat tmp_mat;
             d_cur_pts.download(tmp_mat);
             tmp_mat.copyTo(cv::Mat(1, d_cur_pts.cols, CV_32FC2, &tmp_cur_pts[0]));
             
             d_status.download(tmp_mat);
             tmp_mat.copyTo(cv::Mat(1, d_status.cols, CV_8UC1, &tmp_status[0]));
             
             cur_pts_ = tmp_cur_pts;
             status = tmp_status;
        }

        // Recover KLT-lost tracks via LighterGlue, then drop appearance-drifted tracks,
        // before culling (both no-ops unless their params are enabled).
        recoverLostTracks(status);
        recoverFromKeyframe(status);
        cleanDriftedTracks(status);

        for (int i = 0; i < static_cast<int>(cur_pts_.size()); i++)
            if (status[i] && !inBorder(cur_pts_[i])) status[i] = 0;

        reduceVector(prev_pts_, status);
        reduceVector(cur_pts_, status);
        reduceVector(ids_, status);
        reduceVector(track_cnt_, status);
        ROS_INFO("temporal optical flow costs: %fms", t_o.toc());
    }

    // NOTE: rejectWithF() here hurt accuracy on MH_04 (RANSAC-F at 1px on
    // forward/near-planar motion culls good tracks -> shorter tracks). Left off.

    for (auto &n : track_cnt_) n++;

    // Drop persistent rigid-motion violators (moving objects); no-op unless xfeat_dyn_mask.
    maskDynamicTracks();

    {
        // ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask(); // Keep mask generation on CPU (it involves drawing circles)
        // ROS_DEBUG("set mask costs %fms", t_m.toc());

        // ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = params.max_cnt - static_cast<int>(cur_pts_.size());
        
        // [GPU] 5. Feature Detection
        if (n_max_cnt > 0) {
            if (mask_.empty()) cout << "mask is empty " << endl;
            if (params.xfeat_enable && xfeat_) {
                // Hybrid: seed new features from XFeat keypoints (KLT then tracks them).
                detectNewFeatures(n_max_cnt);
            } else {
                TicToc t_detect;
                // Upload mask to GPU
                d_mask.upload(mask_);

                // Run GPU Detector (Shi-Tomasi)
                // Note: createGoodFeaturesToTrackDetector was set with max_cnt in constructor.
                // If n_max_cnt varies wildly, you might get more points than needed,
                // but we filter them in the loop below anyway.
                gpu_detector->detect(d_cur_img, d_new_pts, d_mask);

                if (!d_new_pts.empty()) {
                    vector<cv::Point2f> n_pts(d_new_pts.cols);
                    cv::Mat n_pts_mat(1, d_new_pts.cols, CV_32FC2, (void*)&n_pts[0]);
                    d_new_pts.download(n_pts_mat);
                    //ROS_INFO("GPU Detection Time: %f ms", t_detect.toc());
                    // Add new points to main list
                    for (auto &p : n_pts) {
                        if (cur_pts_.size() >= params.max_cnt) break; // formatting safety
                        cur_pts_.push_back(p);
                        ids_.push_back(IdCounter::get());
                        track_cnt_.push_back(1);
                    }
                }
            }
        }
        ROS_INFO("detect feature costs: %f ms", t_t.toc());
    }

    // Snapshot a keyframe for wide-baseline recovery (no-op unless xfeat_kf_recover).
    maybeUpdateKeyframe();

    cur_un_pts_ = undistortedPts(cur_pts_, m_camera_[0]);
    pts_velocity_ = ptsVelocity(ids_, cur_un_pts_, cur_un_pts_map_, prev_un_pts_map_);

    // --- STEREO TRACKING (Right Camera) ---
    if (!_img1.empty() && stereo_cam_) {
        ids_right_.clear();
        cur_right_pts_.clear();
        cur_un_right_pts_.clear();
        right_pts_velocity_.clear();
        cur_un_right_pts_map_.clear();
        
        if (!cur_pts_.empty()) {
            // [GPU] 6. Stereo Tracking
            // Track from Left Image (cur_img) -> Right Image (rightImg)
            d_right_img.upload(rightImg);

            // Upload Left Points (cur_pts)
            cv::Mat cur_pts_mat(1, cur_pts_.size(), CV_32FC2, (void *)&cur_pts_[0]);
            d_cur_pts.upload(cur_pts_mat);

            // Track Left -> Right
            gpu_lk_tracker->setUseInitialFlow(false); 
            gpu_lk_tracker->calc(d_cur_img, d_right_img, d_cur_pts, d_cur_pts, d_status, d_err); 
            // Note: output is stored in d_cur_pts temporarily (actually represents right points now)
            
            // We need to store these right points in a different variable for the reverse check
            cv::cuda::GpuMat d_right_pts_found = d_cur_pts.clone(); 

            vector<uchar> status;
            vector<uchar> statusRightLeft;
            
            // Download status for Left->Right
            vector<uchar> tmp_status(d_status.cols);
            cv::Mat status_mat(1, d_status.cols, CV_8UC1, (void*)&tmp_status[0]);
            d_status.download(status_mat);
            status = tmp_status;

            // Reverse Check: Right -> Left
            if (params.flow_back) {
                // Track Right -> Left
                gpu_lk_tracker->calc(d_right_img, d_cur_img, d_right_pts_found, d_reverse_pts, d_reverse_status);
                
                // Download Reverse Status
                vector<uchar> tmp_rev_status(d_reverse_status.cols);
                cv::Mat rev_stat_mat(1, d_reverse_status.cols, CV_8UC1, (void*)&tmp_rev_status[0]);
                d_reverse_status.download(rev_stat_mat);
                statusRightLeft = tmp_rev_status;
                
                // Download Right Points (to check border and distance)
                vector<cv::Point2f> right_pts_cpu(d_right_pts_found.cols);
                cv::Mat r_pts_mat(1, d_right_pts_found.cols, CV_32FC2, (void*)&right_pts_cpu[0]);
                d_right_pts_found.download(r_pts_mat);
                
                // Download Reverse Points (Left guess)
                vector<cv::Point2f> rev_pts_cpu(d_reverse_pts.cols);
                cv::Mat rev_pts_mat(1, d_reverse_pts.cols, CV_32FC2, (void*)&rev_pts_cpu[0]);
                d_reverse_pts.download(rev_pts_mat);

                cur_right_pts_ = right_pts_cpu;

                // Validate
                for (size_t i = 0; i < status.size(); i++) {
                    if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts_[i]) &&
                        distance(cur_pts_[i], rev_pts_cpu[i]) <= 0.5)
                        status[i] = 1;
                    else
                        status[i] = 0;
                }
            }
            
            ids_right_ = ids_;
            reduceVector(cur_right_pts_, status);
            reduceVector(ids_right_, status);

            cur_un_right_pts_ = undistortedPts(cur_right_pts_, m_camera_[1]);
            right_pts_velocity_ = ptsVelocity(ids_right_, cur_un_right_pts_, cur_un_right_pts_map_, prev_un_right_pts_map_);
        }
        prev_un_right_pts_map_ = cur_un_right_pts_map_;
    }

    if (params.show_track)
        drawTrack(cur_img_, rightImg, ids_, cur_pts_, cur_right_pts_, prev_left_pts_map_);

    // [GPU] 7. Save current image to Previous buffer for next loop
    // This is a direct GPU-to-GPU copy, very fast
    d_prev_img = d_cur_img.clone();

    prev_pts_ = cur_pts_;
    prev_un_pts_ = cur_un_pts_;
    prev_un_pts_map_ = cur_un_pts_map_;
    prev_time_ = cur_time_;
    has_prediction_ = false;

    // ... (Remainder of the function is Formatting logic, Keep on CPU) ...
    
    prev_left_pts_map_.clear();
    for (size_t i = 0; i < cur_pts_.size(); i++)
        prev_left_pts_map_[ids_[i]] = cur_pts_[i];

    map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> featureFrame;
    for (size_t i = 0; i < ids_.size(); i++) {
        int feature_id = ids_[i];
        double x = cur_un_pts_[i].x;
        double y = cur_un_pts_[i].y;
        double z = 1;
        double p_u = cur_pts_[i].x;
        double p_v = cur_pts_[i].y;
        int camera_id = 0;
        double velocity_x = pts_velocity_[i].x;
        double velocity_y = pts_velocity_[i].y;

        Eigen::Matrix<double, 8, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }

    if (!_img1.empty() && stereo_cam_) {
        for (size_t i = 0; i < ids_right_.size(); i++) {
            int feature_id = ids_right_[i];
            double x = cur_un_right_pts_[i].x;
            double y = cur_un_right_pts_[i].y;
            double z = 1;
            double p_u = cur_right_pts_[i].x;
            double p_v = cur_right_pts_[i].y;
            int camera_id = 1;
            double velocity_x = right_pts_velocity_[i].x;
            double velocity_y = right_pts_velocity_[i].y;

            Eigen::Matrix<double, 8, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
        }
    }
    printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}

map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>>
FeatureTrackerKLT::trackImageVecCUDA(double _cur_time, const cv::Mat &_img, const cv::Mat &depth,
                           const cv::Mat &_img1) {
    TicToc t_r;
    cur_time_ = _cur_time;
    cur_img_ = _img;
    row = cur_img_.rows;
    col = cur_img_.cols;
    const cv::Mat &rightImg = _img1;
    std::cout << "Tracking image using CUDA!" << std::endl;
    cur_pts_.clear();

    // [GPU] 1. Upload Current Image to GPU
    // We do this ONCE at the start so both tracking and detection can use it
    cv::Mat composite_img = createDualChannelInput(cur_img_, depth_img_);
    cv::cuda::GpuMat d_dual_cur_img;
    d_dual_cur_img.upload(composite_img);
    d_cur_img.upload(cur_img_);

    if (prev_pts_.size() > 0) {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;

        // [GPU] 2. Upload Previous Points
        // We must reshape vector<Point2f> to a 1xN matrix for CUDA
        cv::Mat prev_pts_mat(1, prev_pts_.size(), CV_32FC2, (void *)&prev_pts_[0]);
        d_prev_pts.upload(prev_pts_mat);

        // [GPU] 3. Handle Prediction (IMU guess)
        if (has_prediction_) {
            // If we have a guess, upload it as the "initial flow" for d_cur_pts
            cv::Mat predict_pts_mat(1, predict_pts_.size(), CV_32FC2, (void *)&predict_pts_[0]);
            d_cur_pts.upload(predict_pts_mat);
            
            // Run LK with USE_INITIAL_FLOW flag
            gpu_lk_tracker->setUseInitialFlow(true);
            gpu_lk_tracker->calc(d_prev_img, d_cur_img, d_prev_pts, d_cur_pts, d_status, d_err);
        } else {
            TicToc t_track;
            // No guess, standard tracking
            gpu_lk_tracker->setUseInitialFlow(false);
            gpu_lk_tracker->calc(d_prev_img, d_cur_img, d_prev_pts, d_cur_pts, d_status, d_err);
            ROS_INFO("GPU Tracking Time: %f ms", t_track.toc());
        }

        // [GPU] 4. Forward-Backward Check (Robustness)
        if (params.flow_back) {
            // Track BACKWARDS: Current -> Previous
            // We use the 'd_cur_pts' we just calculated as the starting point
            gpu_lk_tracker->setUseInitialFlow(false); // Usually no prediction for reverse
            gpu_lk_tracker->calc(d_cur_img, d_prev_img, d_cur_pts, d_reverse_pts, d_reverse_status);

            // [GPU -> CPU] Download everything to do the distance check
            // Logic is easier on CPU than writing a custom CUDA kernel
            vector<cv::Point2f> tmp_cur_pts(d_cur_pts.cols);
            vector<cv::Point2f> tmp_rev_pts(d_reverse_pts.cols);
            vector<uchar> tmp_status(d_status.cols);
            vector<uchar> tmp_rev_status(d_reverse_status.cols);

            cv::Mat tmp_mat;
            
            d_cur_pts.download(tmp_mat);
            tmp_mat.copyTo(cv::Mat(1, d_cur_pts.cols, CV_32FC2, &tmp_cur_pts[0]));

            d_reverse_pts.download(tmp_mat);
            tmp_mat.copyTo(cv::Mat(1, d_reverse_pts.cols, CV_32FC2, &tmp_rev_pts[0]));

            d_status.download(tmp_mat);
            tmp_mat.copyTo(cv::Mat(1, d_status.cols, CV_8UC1, &tmp_status[0]));
            
            d_reverse_status.download(tmp_mat);
            tmp_mat.copyTo(cv::Mat(1, d_reverse_status.cols, CV_8UC1, &tmp_rev_status[0]));

            // Verify Tracks
            cur_pts_ = tmp_cur_pts; // Update class member
            status = tmp_status;    // Update local status
            
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i] && tmp_rev_status[i] &&
                    distance(prev_pts_[i], tmp_rev_pts[i]) <= 0.5) {
                    status[i] = 1;
                } else {
                    status[i] = 0;
                }
            }
        }
        else {
             // If no flow_back, just download the forward results
             vector<cv::Point2f> tmp_cur_pts(d_cur_pts.cols);
             vector<uchar> tmp_status(d_status.cols);
             
             cv::Mat tmp_mat;
             d_cur_pts.download(tmp_mat);
             tmp_mat.copyTo(cv::Mat(1, d_cur_pts.cols, CV_32FC2, &tmp_cur_pts[0]));
             
             d_status.download(tmp_mat);
             tmp_mat.copyTo(cv::Mat(1, d_status.cols, CV_8UC1, &tmp_status[0]));
             
             cur_pts_ = tmp_cur_pts;
             status = tmp_status;
        }

        for (int i = 0; i < static_cast<int>(cur_pts_.size()); i++)
            if (status[i] && !inBorder(cur_pts_[i])) status[i] = 0;
            
        reduceVector(prev_pts_, status);
        reduceVector(cur_pts_, status);
        reduceVector(ids_, status);
        reduceVector(track_cnt_, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt_) n++;

    {
        // ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask(); // Keep mask generation on CPU (it involves drawing circles)
        // ROS_DEBUG("set mask costs %fms", t_m.toc());

        // ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = params.max_cnt - static_cast<int>(cur_pts_.size());
        
        // [GPU] 5. Feature Detection
        if (n_max_cnt > 0) {
            if (mask_.empty()) cout << "mask is empty " << endl;
            TicToc t_detect;
            // Upload mask to GPU
            d_mask.upload(mask_);

            // Run GPU Detector (Shi-Tomasi)
            // Note: createGoodFeaturesToTrackDetector was set with max_cnt in constructor.
            // If n_max_cnt varies wildly, you might get more points than needed, 
            // but we filter them in the loop below anyway.
            gpu_detector->detect(d_cur_img, d_new_pts, d_mask);

            if (!d_new_pts.empty()) {
                vector<cv::Point2f> n_pts(d_new_pts.cols);
                cv::Mat n_pts_mat(1, d_new_pts.cols, CV_32FC2, (void*)&n_pts[0]);
                d_new_pts.download(n_pts_mat);
                //ROS_INFO("GPU Detection Time: %f ms", t_detect.toc());
                // Add new points to main list
                for (auto &p : n_pts) {
                    if (cur_pts_.size() >= params.max_cnt) break; // formatting safety
                    cur_pts_.push_back(p);
                    ids_.push_back(IdCounter::get());
                    track_cnt_.push_back(1);
                }
            }
        }
        ROS_DEBUG("detect feature costs: %f ms", t_t.toc());
    }

    cur_un_pts_ = undistortedPts(cur_pts_, m_camera_[0]);
    pts_velocity_ = ptsVelocity(ids_, cur_un_pts_, cur_un_pts_map_, prev_un_pts_map_);

    // --- STEREO TRACKING (Right Camera) ---
    if (!_img1.empty() && stereo_cam_) {
        ids_right_.clear();
        cur_right_pts_.clear();
        cur_un_right_pts_.clear();
        right_pts_velocity_.clear();
        cur_un_right_pts_map_.clear();
        
        if (!cur_pts_.empty()) {
            // [GPU] 6. Stereo Tracking
            // Track from Left Image (cur_img) -> Right Image (rightImg)
            d_right_img.upload(rightImg);

            // Upload Left Points (cur_pts)
            cv::Mat cur_pts_mat(1, cur_pts_.size(), CV_32FC2, (void *)&cur_pts_[0]);
            d_cur_pts.upload(cur_pts_mat);

            // Track Left -> Right
            gpu_lk_tracker->setUseInitialFlow(false); 
            gpu_lk_tracker->calc(d_cur_img, d_right_img, d_cur_pts, d_cur_pts, d_status, d_err); 
            // Note: output is stored in d_cur_pts temporarily (actually represents right points now)
            
            // We need to store these right points in a different variable for the reverse check
            cv::cuda::GpuMat d_right_pts_found = d_cur_pts.clone(); 

            vector<uchar> status;
            vector<uchar> statusRightLeft;
            
            // Download status for Left->Right
            vector<uchar> tmp_status(d_status.cols);
            cv::Mat status_mat(1, d_status.cols, CV_8UC1, (void*)&tmp_status[0]);
            d_status.download(status_mat);
            status = tmp_status;

            // Reverse Check: Right -> Left
            if (params.flow_back) {
                // Track Right -> Left
                gpu_lk_tracker->calc(d_right_img, d_cur_img, d_right_pts_found, d_reverse_pts, d_reverse_status);
                
                // Download Reverse Status
                vector<uchar> tmp_rev_status(d_reverse_status.cols);
                cv::Mat rev_stat_mat(1, d_reverse_status.cols, CV_8UC1, (void*)&tmp_rev_status[0]);
                d_reverse_status.download(rev_stat_mat);
                statusRightLeft = tmp_rev_status;
                
                // Download Right Points (to check border and distance)
                vector<cv::Point2f> right_pts_cpu(d_right_pts_found.cols);
                cv::Mat r_pts_mat(1, d_right_pts_found.cols, CV_32FC2, (void*)&right_pts_cpu[0]);
                d_right_pts_found.download(r_pts_mat);
                
                // Download Reverse Points (Left guess)
                vector<cv::Point2f> rev_pts_cpu(d_reverse_pts.cols);
                cv::Mat rev_pts_mat(1, d_reverse_pts.cols, CV_32FC2, (void*)&rev_pts_cpu[0]);
                d_reverse_pts.download(rev_pts_mat);

                cur_right_pts_ = right_pts_cpu;

                // Validate
                for (size_t i = 0; i < status.size(); i++) {
                    if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts_[i]) &&
                        distance(cur_pts_[i], rev_pts_cpu[i]) <= 0.5)
                        status[i] = 1;
                    else
                        status[i] = 0;
                }
            }
            
            ids_right_ = ids_;
            reduceVector(cur_right_pts_, status);
            reduceVector(ids_right_, status);

            cur_un_right_pts_ = undistortedPts(cur_right_pts_, m_camera_[1]);
            right_pts_velocity_ = ptsVelocity(ids_right_, cur_un_right_pts_, cur_un_right_pts_map_, prev_un_right_pts_map_);
        }
        prev_un_right_pts_map_ = cur_un_right_pts_map_;
    }

    if (params.show_track)
        drawTrack(cur_img_, rightImg, ids_, cur_pts_, cur_right_pts_, prev_left_pts_map_);

    // [GPU] 7. Save current image to Previous buffer for next loop
    // This is a direct GPU-to-GPU copy, very fast
    d_prev_img = d_cur_img.clone();

    prev_pts_ = cur_pts_;
    prev_un_pts_ = cur_un_pts_;
    prev_un_pts_map_ = cur_un_pts_map_;
    prev_time_ = cur_time_;
    has_prediction_ = false;

    // ... (Remainder of the function is Formatting logic, Keep on CPU) ...
    
    prev_left_pts_map_.clear();
    for (size_t i = 0; i < cur_pts_.size(); i++)
        prev_left_pts_map_[ids_[i]] = cur_pts_[i];

    map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> featureFrame;
    for (size_t i = 0; i < ids_.size(); i++) {
        int feature_id = ids_[i];
        double x = cur_un_pts_[i].x;
        double y = cur_un_pts_[i].y;
        double z = 1;
        double p_u = cur_pts_[i].x;
        double p_v = cur_pts_[i].y;
        int camera_id = 0;
        double velocity_x = pts_velocity_[i].x;
        double velocity_y = pts_velocity_[i].y;

        Eigen::Matrix<double, 8, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }

    if (!_img1.empty() && stereo_cam_) {
        for (size_t i = 0; i < ids_right_.size(); i++) {
            int feature_id = ids_right_[i];
            double x = cur_un_right_pts_[i].x;
            double y = cur_un_right_pts_[i].y;
            double z = 1;
            double p_u = cur_right_pts_[i].x;
            double p_v = cur_right_pts_[i].y;
            int camera_id = 1;
            double velocity_x = right_pts_velocity_[i].x;
            double velocity_y = right_pts_velocity_[i].y;

            Eigen::Matrix<double, 8, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
        }
    }
    printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}


void FeatureTrackerKLT::rejectWithF() {
  if (cur_pts_.size() >= 8) {
    ROS_DEBUG("FM ransac begins");
    TicToc t_f;
    vector<cv::Point2f> un_cur_pts(cur_pts_.size());
    vector<cv::Point2f> un_prev_pts(prev_pts_.size());

    for (unsigned i = 0; i < cur_pts_.size(); i++) {
      Eigen::Vector3d tmp_p;
      m_camera_[0]->liftProjective(
          Eigen::Vector2d(cur_pts_[i].x, cur_pts_[i].y), tmp_p);
      tmp_p.x() = params.focal_length * tmp_p.x() / tmp_p.z() + col / 2.0;
      tmp_p.y() = params.focal_length * tmp_p.y() / tmp_p.z() + row / 2.0;
      un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

      m_camera_[0]->liftProjective(
          Eigen::Vector2d(prev_pts_[i].x, prev_pts_[i].y), tmp_p);
      tmp_p.x() = params.focal_length * tmp_p.x() / tmp_p.z() + col / 2.0;
      tmp_p.y() = params.focal_length * tmp_p.y() / tmp_p.z() + row / 2.0;
      un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }

    vector<uchar> status;
    cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC,
                           params.f_threshold, 0.99, status);
    int size_a = cur_pts_.size();
    reduceVector(prev_pts_, status);
    reduceVector(cur_pts_, status);
    // cur_un_pts_ may be stale here (recomputed after detection); only reduce if
    // it currently corresponds to cur_pts_.
    if (cur_un_pts_.size() == status.size()) reduceVector(cur_un_pts_, status);
    reduceVector(ids_, status);
    reduceVector(track_cnt_, status);
    ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts_.size(),
              1.0 * cur_pts_.size() / size_a);
    ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
  }
}

void FeatureTrackerKLT::readIntrinsicParameter(const vector<string> &calib_file) {
  for (size_t i = 0; i < calib_file.size(); i++) {
    ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
    camodocal::CameraPtr camera =
        CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
    m_camera_.push_back(camera);
  }
  if (calib_file.size() == 2) stereo_cam_ = true;

  if (params.xfeat_enable) {
    ROS_INFO("KLT hybrid: seeding new features with XFeat (%s)",
             params.xfeat_engine_path.c_str());
    xfeat_ = std::make_unique<XFeatTRT>(params.xfeat_engine_path);
    // Only pay the dense-map device->host copy when a consumer samples it
    // (semi-dense tracking or dense-sampled cleaning); pure overhead otherwise.
    xfeat_->setDenseEnabled(params.xfeat_semidense || params.xfeat_clean);
    if ((params.xfeat_guided_init || params.xfeat_recover || params.xfeat_kf_recover) &&
        !params.xfeat_lighterglue_engine_path.empty()) {
      ROS_INFO("KLT hybrid: LighterGlue (%s) [guided_init=%d recover=%d kf_recover=%d]",
               params.xfeat_lighterglue_engine_path.c_str(), params.xfeat_guided_init,
               params.xfeat_recover, params.xfeat_kf_recover);
      lighterglue_ =
          std::make_unique<LighterGlueTRT>(params.xfeat_lighterglue_engine_path);
      if (lighterglue_->numKpts() != xfeat_->topK()) {
        std::cerr << "[XFeat] LighterGlue N (" << lighterglue_->numKpts()
                  << ") != XFeat top_k (" << xfeat_->topK()
                  << "); disabling LighterGlue aids." << std::endl;
        lighterglue_.reset();
      }
    }
  }
}

// Run XFeat once for this frame (reused for new-feature seeding) and, if guided
// init is enabled, compute a LighterGlue-homography prediction for KLT.
void FeatureTrackerKLT::extractAndGuide() {
  if (!(params.xfeat_enable && xfeat_)) return;
  // Drop drift anchors for tracks that no longer exist (keep the map bounded).
  if (params.xfeat_clean && !ref_desc_.empty()) {
    std::unordered_set<int> live(ids_.begin(), ids_.end());
    for (auto it = ref_desc_.begin(); it != ref_desc_.end();) {
      if (live.find(it->first) == live.end())
        it = ref_desc_.erase(it);
      else
        ++it;
    }
  }
  // Promote the previous frame's extraction (still in cur_xf_ from last call).
  if (cur_xf_.n > 0) {
    prev_xf_ = std::move(cur_xf_);
    prev_xf_valid_ = true;
  }
  cur_xf_ = xfeat_->run(cur_img_);
  matched_src_.clear();
  matched_dst_.clear();
  // LighterGlue is only needed by guided-init and track-recovery. Descriptor cleaning
  // is XFeat-only (reuses cur_xf_), so skip the matcher entirely when neither is on --
  // the default hybrid (+ optional cleaning) then runs no LighterGlue at all.
  if (lighterglue_ && prev_xf_valid_ &&
      (params.xfeat_guided_init || params.xfeat_recover)) {
    matchPrevCur();  // shared by guided-init and track-recovery
    if (params.xfeat_guided_init && !prev_pts_.empty()) {
      if (computeGuidedPrediction()) has_prediction_ = true;
    }
  }
}

// Confident prev<->cur LighterGlue matches (src=prev kpt, dst=cur kpt).
void FeatureTrackerKLT::matchPrevCur() {
  LGMatches m = lighterglue_->run(prev_xf_.keypoints, prev_xf_.descriptors,
                                  cur_xf_.keypoints, cur_xf_.descriptors);
  matched_src_.reserve(256);
  matched_dst_.reserve(256);
  for (int i = 0; i < m.n; ++i) {
    int j = m.matches0[i];
    if (j < 0 || j >= cur_xf_.n) continue;
    if (m.mscores0[i] < 0.5f) continue;
    matched_src_.push_back(prev_xf_.keypoints[i]);
    matched_dst_.push_back(cur_xf_.keypoints[j]);
  }
}

// Recover tracks KLT lost this frame (status==0) by borrowing the local flow of a
// nearby confident LighterGlue match: cur = q + (dst - src). This applies a REAL
// match's displacement to the track's true previous endpoint q (correct flow, unlike
// snapping to the match's own destination, which injects up to `radius` px of error).
// The borrow is only trusted where the LOCAL flow field is uniform: the nearest
// match's flow is validated against other confident matches within radius, and
// rejected if a majority of neighbours disagree by more than `flow_tol` px (parallax,
// repetitive texture, or wrong-depth -> the constant-flow assumption is invalid). A
// loose per-frame volume cap guards only catastrophic frames.
int FeatureTrackerKLT::recoverLostTracks(std::vector<uchar> &status) {
  if (!params.xfeat_recover || matched_src_.empty()) return 0;
  const float R2 = params.xfeat_recover_radius * params.xfeat_recover_radius;
  const float ftol2 = params.xfeat_recover_flow_tol * params.xfeat_recover_flow_tol;
  struct Cand { size_t k; float d2; cv::Point2f cur; };
  std::vector<Cand> cands;
  for (size_t k = 0; k < status.size() && k < prev_pts_.size(); ++k) {
    if (status[k]) continue;  // KLT already tracks this point well
    const cv::Point2f &q = prev_pts_[k];
    int bi = -1;
    float best = R2;
    for (size_t i = 0; i < matched_src_.size(); ++i) {
      float dx = matched_src_[i].x - q.x, dy = matched_src_[i].y - q.y;
      float d2 = dx * dx + dy * dy;
      if (d2 < best) { best = d2; bi = static_cast<int>(i); }
    }
    if (bi < 0) continue;  // no confident match within radius
    const cv::Point2f f0(matched_dst_[bi].x - matched_src_[bi].x,
                         matched_dst_[bi].y - matched_src_[bi].y);
    // Local flow-consistency check: do neighbouring matches move the same way?
    int total = 0, agree = 0;
    for (size_t i = 0; i < matched_src_.size(); ++i) {
      if (static_cast<int>(i) == bi) continue;
      float dx = matched_src_[i].x - q.x, dy = matched_src_[i].y - q.y;
      if (dx * dx + dy * dy > R2) continue;
      total++;
      float ex = (matched_dst_[i].x - matched_src_[i].x) - f0.x;
      float ey = (matched_dst_[i].y - matched_src_[i].y) - f0.y;
      if (ex * ex + ey * ey <= ftol2) agree++;
    }
    if (total >= 2 && agree * 2 < total) continue;  // nearest flow contradicted -> skip
    cv::Point2f p(q.x + f0.x, q.y + f0.y);
    if (!inBorder(p)) continue;
    cands.push_back({k, best, p});
  }
  if (cands.empty()) return 0;
  // Volume gate: a frame that lost most of its tracks is degraded; mass recovery
  // would flood the estimator with correlated guesses. Keep the closest (best
  // evidence) candidates up to a fraction of the live-track budget.
  size_t cap = static_cast<size_t>(params.xfeat_recover_max_ratio *
                                   static_cast<float>(status.size()));
  if (cap < 1) cap = 1;
  if (cands.size() > cap) {
    std::nth_element(cands.begin(), cands.begin() + cap, cands.end(),
                     [](const Cand &a, const Cand &b) { return a.d2 < b.d2; });
    cands.resize(cap);
  }
  for (const auto &c : cands) {
    cur_pts_[c.k] = c.cur;
    status[c.k] = 1;
  }
  ROS_INFO("XFeat recovered %d KLT-lost tracks (flow-validated)",
           static_cast<int>(cands.size()));
  return static_cast<int>(cands.size());
}

// Drop tracks whose appearance JUMPED this frame -> KLT likely snapped to a wrong
// feature. Each track keeps a reference XFeat descriptor that ROLLS forward every frame;
// we compare it to the descriptor of the nearest CURRENT XFeat keypoint (a proxy for the
// descriptor at the tracked sub-pixel point -- only sparse keypoints exist, not a dense
// map). A large per-frame cosine drop = sudden appearance change = bad jump -> drop;
// otherwise roll the reference to the current descriptor so slow LEGITIMATE drift
// (viewpoint/illumination over a long track) never accumulates into a false drop. (The
// birth-anchor variant did accumulate -> killed good long tracks -> diverged on stairs.)
// Logs cosine min/mean so the threshold can be set from data. Untouched if no keypoint
// within radius (cannot evaluate). Improves track QUALITY, complementary to recovery.
void FeatureTrackerKLT::cleanDriftedTracks(std::vector<uchar> &status) {
  if (!params.xfeat_clean || ref_desc_.empty()) return;
  // With a --dense engine, sample the descriptor at the EXACT tracked sub-pixel
  // location (no sparse nearest-keypoint proxy -> removes the proxy noise that limited
  // cleaning). Otherwise fall back to the nearest sparse keypoint within radius.
  const bool use_dense = (xfeat_ && xfeat_->hasDense());
  if (!use_dense && cur_xf_.n == 0) return;
  const float R2 = params.xfeat_clean_radius * params.xfeat_clean_radius;
  int dropped = 0, evaluated = 0;
  float cmin = 1.0f, csum = 0.0f;
  for (size_t k = 0; k < status.size() && k < cur_pts_.size() && k < ids_.size(); ++k) {
    if (!status[k]) continue;  // already lost this frame; nothing to clean
    auto it = ref_desc_.find(ids_[k]);
    if (it == ref_desc_.end()) continue;  // no reference (e.g. seeded before clean on)
    const cv::Point2f &p = cur_pts_[k];
    std::array<float, 64> dvec;
    if (use_dense) {
      dvec = xfeat_->sampleDense(p.x, p.y);  // exact-location descriptor
    } else {
      int bi = -1;
      float best = R2;
      for (int i = 0; i < cur_xf_.n; ++i) {
        float dx = cur_xf_.keypoints[i].x - p.x, dy = cur_xf_.keypoints[i].y - p.y;
        float d2 = dx * dx + dy * dy;
        if (d2 < best) { best = d2; bi = i; }
      }
      if (bi < 0) continue;  // no current keypoint near the track -> cannot evaluate
      const float *d = &cur_xf_.descriptors[bi * 64];
      for (int c = 0; c < 64; ++c) dvec[c] = d[c];
    }
    float cos = 0.0f;
    for (int c = 0; c < 64; ++c) cos += it->second[c] * dvec[c];  // both unit -> cosine
    evaluated++;
    csum += cos;
    if (cos < cmin) cmin = cos;
    if (cos < params.xfeat_clean_thr) {
      status[k] = 0;  // sudden appearance jump -> KLT snapped to a wrong feature
      dropped++;
    } else {
      it->second = dvec;  // roll reference forward
    }
  }
  if (evaluated > 0)
    ROS_INFO("descriptor-clean[%s]: dropped %d / %d evaluated (cos min=%.2f mean=%.2f)",
             use_dense ? "dense" : "sparse", dropped, evaluated, cmin, csum / evaluated);
}

// Revive KLT-lost tracks by matching the last keyframe against the current frame with
// LighterGlue (wide baseline). For each lost track anchored in the keyframe, look up
// its keyframe keypoint's direct LighterGlue correspondence in the current frame and
// place the track there. No observation gap (alive last frame -> revived this frame).
int FeatureTrackerKLT::recoverFromKeyframe(std::vector<uchar> &status) {
  if (!params.xfeat_kf_recover || !lighterglue_ || kf_xf_.n == 0 ||
      cur_xf_.n == 0 || kf_track_kp_.empty())
    return 0;
  // Skip the matcher entirely on frames with no lost-and-anchored tracks.
  bool any = false;
  for (size_t k = 0; k < status.size() && k < ids_.size(); ++k) {
    if (!status[k] && kf_track_kp_.count(ids_[k])) { any = true; break; }
  }
  if (!any) return 0;
  // Wide-baseline match: keyframe (0) -> current (1). matches0[i] = cur keypoint index.
  LGMatches m = lighterglue_->run(kf_xf_.keypoints, kf_xf_.descriptors,
                                  cur_xf_.keypoints, cur_xf_.descriptors);
  int recovered = 0;
  for (size_t k = 0; k < status.size() && k < ids_.size() && k < cur_pts_.size(); ++k) {
    if (status[k]) continue;  // KLT still tracks it
    auto it = kf_track_kp_.find(ids_[k]);
    if (it == kf_track_kp_.end()) continue;  // track not anchored in the keyframe
    int i = it->second;
    if (i < 0 || i >= m.n) continue;
    if (m.mscores0[i] < params.xfeat_kf_min_conf) continue;
    int j = m.matches0[i];
    if (j < 0 || j >= cur_xf_.n) continue;
    const cv::Point2f &p = cur_xf_.keypoints[j];  // direct wide-baseline correspondence
    if (!inBorder(p)) continue;
    cur_pts_[k] = p;
    status[k] = 1;
    recovered++;
  }
  if (recovered > 0) ROS_INFO("XFeat keyframe-recovered %d KLT-lost tracks", recovered);
  return recovered;
}

// Every xfeat_kf_interval frames, snapshot the current XFeat extraction as the new
// keyframe and anchor each live track to its nearest XFeat keypoint (within radius),
// so it can later be re-found by matching the keyframe against a future frame.
void FeatureTrackerKLT::maybeUpdateKeyframe() {
  if (!params.xfeat_kf_recover || cur_xf_.n == 0) return;
  if (kf_xf_.n != 0 && ++frames_since_kf_ < params.xfeat_kf_interval) return;
  frames_since_kf_ = 0;
  kf_xf_ = cur_xf_;  // deep copy; cur_xf_ is promoted/reused next frame
  kf_track_kp_.clear();
  const float R2 = params.xfeat_kf_anchor_radius * params.xfeat_kf_anchor_radius;
  for (size_t k = 0; k < ids_.size() && k < cur_pts_.size(); ++k) {
    const cv::Point2f &p = cur_pts_[k];
    int bi = -1;
    float best = R2;
    for (int i = 0; i < cur_xf_.n; ++i) {
      float dx = cur_xf_.keypoints[i].x - p.x, dy = cur_xf_.keypoints[i].y - p.y;
      float d2 = dx * dx + dy * dy;
      if (d2 < best) { best = d2; bi = i; }
    }
    if (bi >= 0) kf_track_kp_[ids_[k]] = bi;
  }
}

// Drop features that PERSISTENTLY violate the rigid epipolar constraint (moving objects:
// swaying foliage, water, crowds). Estimate F from all current correspondences (RANSAC);
// a RANSAC outlier moves inconsistently with the dominant rigid motion. One-shot removal
// (rejectWithF) hurt because near-planar/forward motion makes F degenerate and culls good
// static tracks transiently -- so we only drop a track that stays an outlier for
// xfeat_dyn_persist consecutive frames. Must be called while prev_pts_/cur_pts_/ids_ are
// aligned (before detectNewFeatures).
void FeatureTrackerKLT::maskDynamicTracks() {
  if (!params.xfeat_dyn_mask || cur_pts_.size() < 8 ||
      cur_pts_.size() != prev_pts_.size())
    return;
  // Undistort to focal-length pixel coords (same convention as rejectWithF).
  std::vector<cv::Point2f> un_cur(cur_pts_.size()), un_prev(prev_pts_.size());
  for (size_t i = 0; i < cur_pts_.size(); ++i) {
    Eigen::Vector3d p;
    m_camera_[0]->liftProjective(Eigen::Vector2d(cur_pts_[i].x, cur_pts_[i].y), p);
    un_cur[i] = cv::Point2f(params.focal_length * p.x() / p.z() + col / 2.0,
                            params.focal_length * p.y() / p.z() + row / 2.0);
    m_camera_[0]->liftProjective(Eigen::Vector2d(prev_pts_[i].x, prev_pts_[i].y), p);
    un_prev[i] = cv::Point2f(params.focal_length * p.x() / p.z() + col / 2.0,
                             params.focal_length * p.y() / p.z() + row / 2.0);
  }
  std::vector<uchar> inlier;
  cv::findFundamentalMat(un_cur, un_prev, cv::FM_RANSAC, params.xfeat_dyn_thr, 0.99,
                         inlier);
  if (inlier.size() != cur_pts_.size()) return;  // F estimation failed
  std::vector<uchar> keep(cur_pts_.size(), 1);
  int dropped = 0;
  for (size_t k = 0; k < ids_.size(); ++k) {
    int &s = dyn_strikes_[ids_[k]];
    if (inlier[k]) {
      s = 0;
    } else if (++s >= params.xfeat_dyn_persist) {
      keep[k] = 0;
      dropped++;
    }
  }
  if (dropped > 0) {
    reduceVector(prev_pts_, keep);
    reduceVector(cur_pts_, keep);
    reduceVector(ids_, keep);
    reduceVector(track_cnt_, keep);
    if (cur_un_pts_.size() == keep.size()) reduceVector(cur_un_pts_, keep);
    ROS_INFO("dynamic-mask: dropped %d persistent epipolar outliers", dropped);
  }
  // Keep the strike map bounded: drop entries for tracks no longer live.
  std::unordered_set<int> live(ids_.begin(), ids_.end());
  for (auto it = dyn_strikes_.begin(); it != dyn_strikes_.end();)
    it = live.count(it->first) ? std::next(it) : dyn_strikes_.erase(it);
}

// Semi-dense (XFeat-star) tracking: for each active track, search the current dense
// descriptor map in a local window around its predicted position for the best cosine
// match to its rolling reference descriptor (coarse grid + parabolic sub-pixel refine),
// replacing KLT optical flow. Fills cur_pts_/status (aligned to prev_pts_/ids_).
void FeatureTrackerKLT::trackDense(std::vector<uchar> &status) {
  const size_t n = prev_pts_.size();
  status.assign(n, 0);
  cur_pts_.assign(n, cv::Point2f(0.f, 0.f));
  if (!xfeat_ || !xfeat_->hasDense()) return;
  const float R = params.xfeat_sd_radius;
  const float step = std::max(1.0f, params.xfeat_sd_step);
  const float thr = params.xfeat_sd_thr;
  auto cosTo = [&](const std::array<float, 64> &r, float x, float y) {
    std::array<float, 64> d = xfeat_->sampleDense(x, y);
    float c = 0.f;
    for (int i = 0; i < 64; ++i) c += r[i] * d[i];
    return c;
  };
  int tracked = 0;
  for (size_t k = 0; k < n; ++k) {
    auto it = sd_ref_.find(ids_[k]);
    if (it == sd_ref_.end()) continue;  // no reference descriptor
    const std::array<float, 64> &r = it->second;
    cv::Point2f c0 = (has_prediction_ && k < predict_pts_.size()) ? predict_pts_[k]
                                                                  : prev_pts_[k];
    // Coarse local grid search for the best descriptor match.
    float best = -2.f, bx = c0.x, by = c0.y;
    for (float dy = -R; dy <= R; dy += step) {
      for (float dx = -R; dx <= R; dx += step) {
        float x = c0.x + dx, y = c0.y + dy;
        if (!inBorder(cv::Point2f(x, y))) continue;
        float cval = cosTo(r, x, y);
        if (cval > best) { best = cval; bx = x; by = y; }
      }
    }
    if (best < thr) continue;  // no confident match -> lost
    // Parabolic sub-pixel refine on the cosine surface (x, y independently).
    float cxm = cosTo(r, bx - step, by), cxp = cosTo(r, bx + step, by);
    float cym = cosTo(r, bx, by - step), cyp = cosTo(r, bx, by + step);
    float dnx = cxm - 2.f * best + cxp, dny = cym - 2.f * best + cyp;
    float ox = (dnx < -1e-6f) ? 0.5f * (cxm - cxp) / dnx : 0.f;
    float oy = (dny < -1e-6f) ? 0.5f * (cym - cyp) / dny : 0.f;
    ox = std::max(-1.f, std::min(1.f, ox));
    oy = std::max(-1.f, std::min(1.f, oy));
    cv::Point2f p(bx + ox * step, by + oy * step);
    if (!inBorder(p)) p = cv::Point2f(bx, by);
    cur_pts_[k] = p;
    status[k] = 1;
    tracked++;
    it->second = xfeat_->sampleDense(p.x, p.y);  // roll reference to matched location
  }
  // Prune the reference map to live ids.
  std::unordered_set<int> live(ids_.begin(), ids_.end());
  for (auto i = sd_ref_.begin(); i != sd_ref_.end();)
    i = live.count(i->first) ? std::next(i) : sd_ref_.erase(i);
  ROS_INFO("semi-dense: tracked %d / %zu", tracked, n);
}

// LighterGlue-match prev<->cur XFeat, fit a RANSAC homography, warp prev_pts_ into
// predict_pts_ (aligned to prev_pts_). Returns false if not enough matches.
bool FeatureTrackerKLT::computeGuidedPrediction() {
  const std::vector<cv::Point2f> &src = matched_src_;  // filled by matchPrevCur()
  const std::vector<cv::Point2f> &dst = matched_dst_;
  if (src.size() < 12) return false;

  // Large-displacement gate: only guide when inter-frame motion is large. For small
  // motion, pyramidal KLT-from-prev is already accurate and a guess only adds noise
  // (un-gated guided init was a wash). Use the median confident-match displacement as
  // a robust per-frame motion measure.
  if (params.xfeat_guided_min_disp > 0.0f) {
    std::vector<float> disp;
    disp.reserve(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
      float dx = dst[i].x - src[i].x, dy = dst[i].y - src[i].y;
      disp.push_back(std::sqrt(dx * dx + dy * dy));
    }
    std::nth_element(disp.begin(), disp.begin() + disp.size() / 2, disp.end());
    if (disp[disp.size() / 2] < params.xfeat_guided_min_disp)
      return false;  // small motion -> let KLT run unguided
  }
  ROS_INFO("guided-init: applying KLT init guess (large-motion frame)");

  predict_pts_.resize(prev_pts_.size());
  predict_pts_debug_.clear();

  if (params.xfeat_guided_init == 2) {
    // Per-point: each tracked point inherits the displacement of the nearest
    // confident match (parallax-aware; falls back to prev location if no match
    // within radius). Better than a global homography for 3-D scenes.
    const float r2 = 80.0f * 80.0f;
    for (size_t k = 0; k < prev_pts_.size(); ++k) {
      const cv::Point2f &q = prev_pts_[k];
      float best = r2;
      int bi = -1;
      for (size_t i = 0; i < src.size(); ++i) {
        float dx = src[i].x - q.x, dy = src[i].y - q.y;
        float d2 = dx * dx + dy * dy;
        if (d2 < best) { best = d2; bi = static_cast<int>(i); }
      }
      predict_pts_[k] = (bi >= 0) ? cv::Point2f(q.x + (dst[bi].x - src[bi].x),
                                                q.y + (dst[bi].y - src[bi].y))
                                  : q;
      predict_pts_debug_.push_back(predict_pts_[k]);
    }
    return true;
  }

  // Mode 1: global RANSAC homography warp.
  cv::Mat H = cv::findHomography(src, dst, cv::RANSAC, 3.0);
  if (H.empty()) return false;
  std::vector<cv::Point2f> warped;
  cv::perspectiveTransform(prev_pts_, warped, H);
  for (size_t k = 0; k < prev_pts_.size(); ++k) {
    const cv::Point2f &p = warped[k];
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < -50 || p.x > col + 50 ||
        p.y < -50 || p.y > row + 50) {
      predict_pts_[k] = prev_pts_[k];
    } else {
      predict_pts_[k] = p;
    }
    predict_pts_debug_.push_back(predict_pts_[k]);
  }
  return true;
}

// Seed up to (max_cnt - currently tracked) new features from XFeat keypoints,
// highest score first, respecting the current mask_ (min_dist spacing). KLT
// optical flow then tracks them across subsequent frames.
void FeatureTrackerKLT::detectNewFeatures(int n_max_cnt) {
  if (n_max_cnt <= 0 || !xfeat_) return;
  if (cur_xf_.n == 0) cur_xf_ = xfeat_->run(cur_img_);  // normally extracted at frame start
  const XFeatFeatures &xf = cur_xf_;
  std::vector<int> order(xf.n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return xf.scores[a] > xf.scores[b]; });
  std::vector<cv::Point2f> new_pts;
  std::vector<int> new_src;  // cur_xf_ keypoint index each seed came from (anchor descriptor)
  // One seeding pass: take keypoints (highest score first) with score >= score_floor,
  // respecting the mask; each accepted seed blocks a `spacing`-radius disc so later
  // seeds stay spread out.
  auto seed = [&](float score_floor, int spacing) {
    for (int idx : order) {
      if (static_cast<int>(cur_pts_.size() + new_pts.size()) >= params.max_cnt) break;
      if (xf.scores[idx] < score_floor) break;  // sorted: rest are lower
      const cv::Point2f &p = xf.keypoints[idx];
      int xi = cvRound(p.x), yi = cvRound(p.y);
      if (xi < 0 || xi >= col || yi < 0 || yi >= row) continue;
      if (!mask_.empty() && mask_.at<uchar>(yi, xi) != 255) continue;
      new_pts.push_back(p);
      new_src.push_back(idx);
      cv::circle(mask_, p, spacing, 0, -1);
    }
  };
  seed(params.xfeat_score_thr, params.min_dist);  // pass 1: best keypoints, full spacing
  // Adaptive density: if pass 1 under-filled the budget (low-texture -> too few strong,
  // well-spread keypoints), top up with weaker keypoints packed at reduced spacing.
  // Texture-rich frames hit max_cnt in pass 1, so this is a no-op there.
  if (params.xfeat_adaptive_density &&
      static_cast<int>(cur_pts_.size() + new_pts.size()) < params.max_cnt) {
    size_t before = new_pts.size();
    int spacing = std::max(
        1, static_cast<int>(params.min_dist * params.xfeat_adaptive_min_ratio));
    seed(params.xfeat_score_thr * params.xfeat_adaptive_score_ratio, spacing);
    if (new_pts.size() > before)
      ROS_INFO("adaptive-density: +%d weak fill (low-texture frame); total %d",
               static_cast<int>(new_pts.size() - before),
               static_cast<int>(cur_pts_.size() + new_pts.size()));
  }
  // Snap seeds to sub-pixel corners so KLT starts on well-localized, trackable
  // points (XFeat keypoints sit on the 8x8 detection grid).
  if (params.xfeat_subpix && !new_pts.empty()) {
    cv::cornerSubPix(
        cur_img_, new_pts, cv::Size(5, 5), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 0.01));
  }
  for (size_t j = 0; j < new_pts.size(); ++j) {
    cur_pts_.push_back(new_pts[j]);
    int id = IdCounter::get();
    ids_.push_back(id);
    track_cnt_.push_back(1);
    // Seed this track's rolling reference with its birth descriptor (XFeat is L2-normalized).
    if (params.xfeat_clean) {
      const float *d = &xf.descriptors[new_src[j] * 64];
      std::array<float, 64> a;
      for (int c = 0; c < 64; ++c) a[c] = d[c];
      ref_desc_[id] = a;
    }
    // Semi-dense: seed the rolling reference from the dense map at the seed location
    // (consistent with how trackDense samples it).
    if (params.xfeat_semidense && xfeat_ && xfeat_->hasDense())
      sd_ref_[id] = xfeat_->sampleDense(new_pts[j].x, new_pts[j].y);
  }
}

void FeatureTrackerKLT::showUndistortion(const string & /*name*/) {
  cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
  vector<Eigen::Vector2d> distortedp;
  vector<Eigen::Vector2d> undistortedp;

  for (int i = 0; i < col; i++)
    for (int j = 0; j < row; j++) {
      Eigen::Vector2d a(i, j);
      Eigen::Vector3d b;
      m_camera_[0]->liftProjective(a, b);
      distortedp.push_back(a);
      undistortedp.emplace_back(b.x() / b.z(), b.y() / b.z());
      // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
    }

  for (int i = 0; i < static_cast<int>(undistortedp.size()); i++) {
    cv::Mat pp(3, 1, CV_32FC1);
    pp.at<float>(0, 0) = undistortedp[i].x() * params.focal_length + col / 2;
    pp.at<float>(1, 0) = undistortedp[i].y() * params.focal_length + row / 2;
    pp.at<float>(2, 0) = 1.0;
    // cout << trackerData[0].K << endl;
    // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
    // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
    if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 &&
        pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600) {
      undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300,
                               pp.at<float>(0, 0) + 300) =
          cur_img_.at<uchar>(distortedp[i].y(), distortedp[i].x());
    } else {
      // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x,
      // pp.at<float>(1, 0), pp.at<float>(0, 0));
    }
  }
  // turn the following code on if you need
  // cv::imshow(name, undistortedImg);
  // cv::waitKey(0);
}

vector<cv::Point2f> FeatureTrackerKLT::undistortedPts(
    vector<cv::Point2f> &pts, const camodocal::CameraPtr &cam) {
  vector<cv::Point2f> un_pts;
  for (auto &pt : pts) {
    Eigen::Vector2d a(pt.x, pt.y);
    Eigen::Vector3d b;
    cam->liftProjective(a, b);
    un_pts.emplace_back(b.x() / b.z(), b.y() / b.z());
  }
  return un_pts;
}

vector<cv::Point2f> FeatureTrackerKLT::ptsVelocity(
    vector<int> &ids, vector<cv::Point2f> &pts,
    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts) {
  vector<cv::Point2f> pts_velocity;
  cur_id_pts.clear();
  for (unsigned i = 0; i < ids.size(); i++) {
    cur_id_pts.insert(make_pair(ids[i], pts[i]));
  }

  // caculate points velocity
  if (!prev_id_pts.empty()) {
    double dt = cur_time_ - prev_time_;

    for (unsigned i = 0; i < pts.size(); i++) {
      std::map<int, cv::Point2f>::iterator it;
      it = prev_id_pts.find(ids[i]);
      if (it != prev_id_pts.end()) {
        double v_x = (pts[i].x - it->second.x) / dt;
        double v_y = (pts[i].y - it->second.y) / dt;
        pts_velocity.emplace_back(v_x, v_y);
      } else
        pts_velocity.emplace_back(0, 0);
    }
  } else {
    for (unsigned i = 0; i < cur_pts_.size(); i++) {
      pts_velocity.emplace_back(0, 0);
    }
  }
  return pts_velocity;
}

void FeatureTrackerKLT::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts,
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap) {
  // int rows = imLeft.rows;
  int cols = imLeft.cols;
  if (!imRight.empty() && stereo_cam_)
    cv::hconcat(imLeft, imRight, im_track_);
  else
    im_track_ = imLeft.clone();
  cv::cvtColor(im_track_, im_track_, cv::COLOR_GRAY2RGB);

  for (size_t j = 0; j < curLeftPts.size(); j++) {
    double len = std::min(1.0, 1.0 * track_cnt_[j] / 20);
    cv::circle(im_track_, curLeftPts[j], 2,
               cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
  }
  if (!imRight.empty() && stereo_cam_) {
    for (size_t i = 0; i < curRightPts.size(); i++) {
      cv::Point2f rightPt = curRightPts[i];
      rightPt.x += cols;
      cv::circle(im_track_, rightPt, 2, cv::Scalar(0, 255, 0), 2);
      // cv::Point2f leftPt = curLeftPtsTrackRight[i];
      // cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
    }
  }

  map<int, cv::Point2f>::iterator mapIt;
  for (size_t i = 0; i < curLeftIds.size(); i++) {
    int id = curLeftIds[i];
    mapIt = prevLeftPtsMap.find(id);
    if (mapIt != prevLeftPtsMap.end()) {
      cv::arrowedLine(im_track_, curLeftPts[i], mapIt->second,
                      cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }
  }

  // draw prediction
  /*
  for(size_t i = 0; i < predict_pts_debug.size(); i++)
  {
      cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255),
  2);
  }
  */
  // printf("predict pts size %d \n", (int)predict_pts_debug.size());

  // cv::Mat imCur2Compress;
  // cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}
void FeatureTrackerKLT::updateDepth(const cv::Mat &depthImg) {
  depth_img_ = depthImg;
  ready = true;
}

void FeatureTrackerKLT::drawDepthTrack(const cv::Mat &imLeft,
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts,
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap) {
  // int rows = imLeft.rows;
  int cols = imLeft.cols;
  d_track = imLeft.clone();
  cv::cvtColor(d_track, d_track, cv::COLOR_GRAY2RGB);

  for (size_t j = 0; j < curLeftPts.size(); j++) {
    double len = std::min(1.0, 1.0 * track_cnt_[j] / 20);
    cv::circle(d_track, curLeftPts[j], 2,
               cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
  }
 

  map<int, cv::Point2f>::iterator mapIt;
  for (size_t i = 0; i < curLeftIds.size(); i++) {
    int id = curLeftIds[i];
    mapIt = prevLeftPtsMap.find(id);
    if (mapIt != prevLeftPtsMap.end()) {
      cv::arrowedLine(d_track, curLeftPts[i], mapIt->second,
                      cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }
  }

  // draw prediction
  /*
  for(size_t i = 0; i < predict_pts_debug.size(); i++)
  {
      cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255),
  2);
  }
  */
  // printf("predict pts size %d \n", (int)predict_pts_debug.size());

  // cv::Mat imCur2Compress;
  // cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

void FeatureTrackerKLT::setPrediction(map<int, Eigen::Vector3d> &predictPts) {
  has_prediction_ = true;
  predict_pts_.clear();
  predict_pts_debug_.clear();
  map<int, Eigen::Vector3d>::iterator itPredict;
  for (size_t i = 0; i < ids_.size(); i++) {
    // printf("prevLeftId size %d prevLeftPts size
    // %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
    int id = ids_[i];
    itPredict = predictPts.find(id);
    if (itPredict != predictPts.end()) {
      Eigen::Vector2d tmp_uv;
      m_camera_[0]->spaceToPlane(itPredict->second, tmp_uv);
      predict_pts_.emplace_back(tmp_uv.x(), tmp_uv.y());
      predict_pts_debug_.emplace_back(tmp_uv.x(), tmp_uv.y());
    } else
      predict_pts_.push_back(prev_pts_[i]);
  }
}

void FeatureTrackerKLT::removeOutliers(set<int> &removePtsIds) {
  std::set<int>::iterator itSet;
  vector<uchar> status;
  for (size_t i = 0; i < ids_.size(); i++) {
    itSet = removePtsIds.find(ids_[i]);
    if (itSet != removePtsIds.end())
      status.push_back(0);
    else
      status.push_back(1);
  }

  reduceVector(prev_pts_, status);
  reduceVector(ids_, status);
  reduceVector(track_cnt_, status);
}

cv::Mat FeatureTrackerKLT::getTrackImage() { return im_track_; }
cv::Mat FeatureTrackerKLT::getDepthTrackImage() { return d_track; }
}  // namespace vins::estimator