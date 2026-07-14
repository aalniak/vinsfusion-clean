#pragma once

#include <camodocal/camera_models/CameraFactory.h>
#include <camodocal/camera_models/CataCamera.h>
#include <camodocal/camera_models/PinholeCamera.h>
#include <vins_estimator/estimator/parameters.h>
#include <vins_estimator/featureTracker/id_counter.h>
#include <vins_estimator/featureTracker/lighterglue_trt.h>
#include <vins_estimator/featureTracker/xfeat_trt.h>
#include <vins_estimator/utility/tic_toc.h>

#include <eigen3/Eigen/Dense>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>

namespace vins::estimator {

// Mono feature tracker that uses XFeat (detect+describe) + LighterGlue (match) as a
// drop-in replacement for the KLT front-end. Each frame is extracted once; the
// previous frame's descriptors are matched to the current frame and feature IDs are
// propagated along matches, with new IDs minted (min_dist-spread, up to max_cnt) for
// unmatched keypoints. Output is the canonical VINS feature frame.
class FeatureTrackerXFeat {
 public:
  explicit FeatureTrackerXFeat(Parameters &params);

  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 8, 1>>>> trackImage(
      double cur_time, const cv::Mat &img, const cv::Mat &img1 = cv::Mat());

  void readIntrinsicParameter(const std::vector<std::string> &calib_file);

  static void setPrediction(std::map<int, Eigen::Vector3d> & /*predictPts*/) {}
  void removeOutliers(std::set<int> &removePtsIds);
  cv::Mat getTrackImage();

 private:
  // Mutual-nearest-neighbour cosine matcher over L2-normalized descriptors,
  // used when params.xfeat_matcher == MNN. Returns matches0 like LighterGlue.
  LGMatches matchMNN(const std::vector<float> &desc0,
                     const std::vector<float> &desc1, int n);

  static std::vector<cv::Point2f> undistortedPts(
      const std::vector<cv::Point2f> &pts, const camodocal::CameraPtr &cam);
  std::vector<cv::Point2f> ptsVelocity(const std::vector<int> &ids,
                                       const std::vector<cv::Point2f> &un_pts,
                                       std::map<int, cv::Point2f> &cur_id_pts,
                                       std::map<int, cv::Point2f> &prev_id_pts,
                                       double dt);
  void drawTrack(const cv::Mat &img, const std::vector<int> &ids,
                 const std::vector<cv::Point2f> &pts,
                 const std::vector<int> &track_cnt,
                 std::map<int, cv::Point2f> &prev_pts_map);
  bool inBorder(const cv::Point2f &pt) const;

  Parameters &params;
  std::unique_ptr<XFeatTRT> xfeat_;
  std::unique_ptr<LighterGlueTRT> lighterglue_;
  std::vector<camodocal::CameraPtr> m_camera_;

  int row_ = 0, col_ = 0;
  int n_ = 0;  // engine top_k

  // Previous-frame state, indexed by extractor slot [0, n_).
  std::vector<cv::Point2f> prev_kpts_;     // size n_ (raw pixel coords)
  std::vector<float> prev_desc_;           // size n_*64
  std::vector<int> prev_id_;               // feature id per slot, or -1
  std::vector<int> prev_track_cnt_;        // track length per slot
  bool has_prev_ = false;
  double prev_time_ = 0.0;

  std::map<int, cv::Point2f> prev_un_pts_map_;  // id -> undistorted pt (for velocity)
  std::map<int, cv::Point2f> prev_pts_map_;     // id -> pixel pt (for drawing)
  cv::Mat im_track_;
};

}  // namespace vins::estimator
