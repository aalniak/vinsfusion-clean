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

#pragma once

#include <camodocal/camera_models/CameraFactory.h>
#include <camodocal/camera_models/CataCamera.h>
#include <camodocal/camera_models/PinholeCamera.h>
#include <execinfo.h>
#include <vins_estimator/estimator/parameters.h>
#include <vins_estimator/utility/tic_toc.h>
#include <vins_estimator/featureTracker/id_counter.h>
#include <vins_estimator/featureTracker/xfeat_trt.h>
#include <vins_estimator/featureTracker/lighterglue_trt.h>
#include <memory>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <csignal>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace camodocal;
using namespace Eigen;

namespace vins::estimator {

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTrackerKLT {
 public:
  explicit FeatureTrackerKLT(Parameters &params);
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(
      double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImageCUDA(
      double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImageVecCUDA(
    double _cur_time, const cv::Mat &_img, const cv::Mat &depth, const cv::Mat &_img1 = cv::Mat());
  void readIntrinsicParameter(const vector<string> &calib_file);
  void setPrediction(map<int, Eigen::Vector3d> &predictPts);
  void removeOutliers(set<int> &removePtsIds);
  cv::Mat getTrackImage();
  cv::Mat getDepthTrackImage();
  void updateDepth(const cv::Mat &depthImg);
 private:
  void setMask();
  void showUndistortion(const string &name);
  void rejectWithF();
  static vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts,
                                            const camodocal::CameraPtr &cam);
  vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                  map<int, cv::Point2f> &cur_id_pts,
                                  map<int, cv::Point2f> &prev_id_pts);
  void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                 vector<int> &curLeftIds, vector<cv::Point2f> &curLeftPts,
                 vector<cv::Point2f> &curRightPts,
                 map<int, cv::Point2f> &prevLeftPtsMap);
  bool inBorder(const cv::Point2f &pt) const;
  static double distance(const cv::Point2f &pt1, const cv::Point2f &pt2);
  void drawDepthTrack(const cv::Mat &imLeft,
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts,
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap);
  
  Parameters &params;

  int row, col;
  cv::Mat im_track_;
  cv::Mat d_track;
  cv::Mat depth_img_;
  cv::Mat mask_;
  cv::Mat fisheye_mask_;
  cv::Mat prev_img_, cur_img_;
    //add gpu-specific items
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> gpu_lk_tracker;
  cv::Ptr<cv::cuda::CornersDetector> gpu_detector;
  
  cv::cuda::GpuMat d_prev_img, d_cur_img, d_right_img;
  cv::cuda::GpuMat d_prev_pts, d_cur_pts, d_status, d_err;
  cv::cuda::GpuMat d_reverse_pts, d_reverse_status;
  cv::cuda::GpuMat d_mask;
  cv::cuda::GpuMat d_new_pts; // For feature detection
  
  // No upload/download, shared memory
  cv::cuda::HostMem mem_cur_img;      // Buffer for current image
  cv::cuda::HostMem mem_prev_pts;     // Buffer for sending points to GPU
  cv::cuda::HostMem mem_cur_pts;      // Buffer for receiving points from GPU
  cv::cuda::HostMem mem_status;       // Buffer for status
  cv::cuda::HostMem mem_err;          // Buffer for error
  cv::cuda::HostMem mem_reverse_pts;
  cv::cuda::HostMem mem_reverse_status;
  cv::Mat cpu_cur_img_view;           // CPU way to see image
  cv::cuda::GpuMat gpu_cur_img_view;  // GPU way to see image

  vector<cv::Point2f> predict_pts_;
  vector<cv::Point2f> predict_pts_debug_;
  vector<cv::Point2f> prev_pts_, cur_pts_, cur_right_pts_;
  vector<cv::Point2f> prev_un_pts_, cur_un_pts_, cur_un_right_pts_;
  vector<cv::Point2f> pts_velocity_, right_pts_velocity_;
  vector<int> ids_, ids_right_;
  vector<int> track_cnt_;
  map<int, cv::Point2f> cur_un_pts_map_, prev_un_pts_map_;
  map<int, cv::Point2f> cur_un_right_pts_map_, prev_un_right_pts_map_;
  map<int, cv::Point2f> prev_left_pts_map_;
  vector<camodocal::CameraPtr> m_camera_;
  double cur_time_;
  double prev_time_;
  bool stereo_cam_;
  bool has_prediction_;

  // Hybrid mode: when params.xfeat_enable, new features are seeded from XFeat
  // keypoints (robust, learned) instead of Shi-Tomasi, then tracked by KLT
  // optical flow (long, continuous tracks for depth). Null when disabled.
  std::unique_ptr<XFeatTRT> xfeat_;
  void detectNewFeatures(int n_max_cnt);

  // Guided initialization (params.xfeat_guided_init): match prev<->cur XFeat with
  // LighterGlue, fit a RANSAC homography, and warp prev_pts_ into predict_pts_ as
  // KLT's initial-flow guess — eases KLT under large displacement / motion blur.
  std::unique_ptr<LighterGlueTRT> lighterglue_;
  XFeatFeatures cur_xf_;     // current frame XFeat (extracted once, reused for seeding)
  XFeatFeatures prev_xf_;    // previous frame XFeat (for guided matching)
  bool prev_xf_valid_ = false;
  // Confident prev<->cur LighterGlue matches for this frame (src=prev, dst=cur),
  // computed once and shared by guided-init and track-recovery.
  std::vector<cv::Point2f> matched_src_, matched_dst_;
  void extractAndGuide();    // run XFeat (+ optional guided prediction) at frame start
  void matchPrevCur();       // fill matched_src_/matched_dst_ via LighterGlue
  bool computeGuidedPrediction();
  // Recover KLT-lost tracks (status==0) using a nearby confident LighterGlue
  // match's displacement. Only touches failures -> cannot regress good tracks.
  int recoverLostTracks(std::vector<uchar> &status);

  // Descriptor-consistency cleaning (params.xfeat_clean): each track keeps a reference
  // XFeat descriptor that ROLLS forward each frame (nearest keypoint's descriptor); a
  // large per-frame cosine drop = KLT snapped to a wrong feature -> drop the track.
  // Rolling (vs a fixed birth anchor) tolerates slow legitimate appearance change --
  // the anchor variant accumulated drift, killed good long tracks, and diverged on
  // stairs. Keyed by feature id (survives reduceVector); pruned in extractAndGuide().
  std::unordered_map<int, std::array<float, 64>> ref_desc_;
  void cleanDriftedTracks(std::vector<uchar> &status);
};

}  // namespace vins::estimator
