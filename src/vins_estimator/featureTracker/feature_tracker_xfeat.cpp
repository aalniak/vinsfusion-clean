#include "vins_estimator/featureTracker/feature_tracker_xfeat.h"

#include <ros/ros.h>

#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <numeric>

namespace vins::estimator {

namespace {
constexpr int BORDER_SIZE = 1;
}  // namespace

FeatureTrackerXFeat::FeatureTrackerXFeat(Parameters &params) : params(params) {}

bool FeatureTrackerXFeat::inBorder(const cv::Point2f &pt) const {
  int x = cvRound(pt.x);
  int y = cvRound(pt.y);
  return BORDER_SIZE <= x && x < col_ - BORDER_SIZE && BORDER_SIZE <= y &&
         y < row_ - BORDER_SIZE;
}

void FeatureTrackerXFeat::readIntrinsicParameter(
    const std::vector<std::string> &calib_file) {
  for (const auto &f : calib_file) {
    ROS_INFO("XFeat: reading camera param %s", f.c_str());
    m_camera_.push_back(
        camodocal::CameraFactory::instance()->generateCameraFromYamlFile(f));
  }
  col_ = m_camera_[0]->imageWidth();
  row_ = m_camera_[0]->imageHeight();

  xfeat_ = std::make_unique<XFeatTRT>(params.xfeat_engine_path);
  lighterglue_ =
      std::make_unique<LighterGlueTRT>(params.xfeat_lighterglue_engine_path);
  n_ = xfeat_->topK();
  if (lighterglue_->numKpts() != n_) {
    std::cerr << "[XFeat] engine top_k (" << n_ << ") != LighterGlue N ("
              << lighterglue_->numKpts() << "); rebuild engines to match."
              << std::endl;
    std::abort();
  }
  prev_id_.assign(n_, -1);
  prev_track_cnt_.assign(n_, 0);
}

LGMatches FeatureTrackerXFeat::matchMNN(const std::vector<float> &desc0,
                                        const std::vector<float> &desc1, int n) {
  using RM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<const RM> D0(desc0.data(), n, 64);
  Eigen::Map<const RM> D1(desc1.data(), n, 64);
  RM sim = D0 * D1.transpose();  // cosine (descriptors are L2-normalized)

  LGMatches out;
  out.n = n;
  out.matches0.assign(n, -1);
  out.mscores0.assign(n, 0.f);
  std::vector<int> argmax_col(n);
  for (int j = 0; j < n; ++j) {
    int bi = 0;
    sim.col(j).maxCoeff(&bi);
    argmax_col[j] = bi;
  }
  for (int i = 0; i < n; ++i) {
    int bj = 0;
    float s = sim.row(i).maxCoeff(&bj);
    if (argmax_col[bj] == i) {  // mutual nearest neighbour
      out.matches0[i] = bj;
      out.mscores0[i] = s;
    }
  }
  return out;
}

std::vector<cv::Point2f> FeatureTrackerXFeat::undistortedPts(
    const std::vector<cv::Point2f> &pts, const camodocal::CameraPtr &cam) {
  std::vector<cv::Point2f> un;
  un.reserve(pts.size());
  for (const auto &p : pts) {
    Eigen::Vector2d a(p.x, p.y);
    Eigen::Vector3d b;
    cam->liftProjective(a, b);
    un.emplace_back(b.x() / b.z(), b.y() / b.z());
  }
  return un;
}

std::vector<cv::Point2f> FeatureTrackerXFeat::ptsVelocity(
    const std::vector<int> &ids, const std::vector<cv::Point2f> &un_pts,
    std::map<int, cv::Point2f> &cur_id_pts, std::map<int, cv::Point2f> &prev_id_pts,
    double dt) {
  std::vector<cv::Point2f> velocity;
  cur_id_pts.clear();
  for (size_t i = 0; i < ids.size(); ++i) cur_id_pts[ids[i]] = un_pts[i];

  velocity.reserve(un_pts.size());
  if (!prev_id_pts.empty() && dt > 1e-9) {
    for (size_t i = 0; i < un_pts.size(); ++i) {
      auto it = prev_id_pts.find(ids[i]);
      if (it != prev_id_pts.end()) {
        velocity.emplace_back((un_pts[i].x - it->second.x) / dt,
                              (un_pts[i].y - it->second.y) / dt);
      } else {
        velocity.emplace_back(0, 0);
      }
    }
  } else {
    velocity.assign(un_pts.size(), cv::Point2f(0, 0));
  }
  return velocity;
}

void FeatureTrackerXFeat::drawTrack(const cv::Mat &img, const std::vector<int> &ids,
                                    const std::vector<cv::Point2f> &pts,
                                    const std::vector<int> &track_cnt,
                                    std::map<int, cv::Point2f> &prev_pts_map) {
  if (img.channels() == 3) {
    im_track_ = img.clone();
  } else {
    cv::cvtColor(img, im_track_, cv::COLOR_GRAY2RGB);
  }
  for (size_t j = 0; j < pts.size(); ++j) {
    double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
    cv::circle(im_track_, pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    auto it = prev_pts_map.find(ids[j]);
    if (it != prev_pts_map.end()) {
      cv::arrowedLine(im_track_, pts[j], it->second, cv::Scalar(0, 255, 0), 1, 8, 0,
                      0.2);
    }
  }
}

std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 8, 1>>>>
FeatureTrackerXFeat::trackImage(double cur_time, const cv::Mat &img,
                                const cv::Mat & /*img1*/) {
  TicToc t_all;
  TicToc t_step;
  col_ = img.cols;
  row_ = img.rows;

  // 1. Extract.
  XFeatFeatures cur = xfeat_->run(img);
  double t_extract = t_step.toc();
  t_step.tic();

  // 2. Match to previous frame and inherit IDs along matches.
  int n_prev_tracks = 0;
  for (int i = 0; i < n_; ++i)
    if (prev_id_[i] >= 0) n_prev_tracks++;
  std::vector<int> cur_id(n_, -1);
  std::vector<int> cur_cnt(n_, 0);
  std::vector<float> best(n_, -1.f);
  if (has_prev_) {
    LGMatches m =
        (params.xfeat_matcher == 1)
            ? matchMNN(prev_desc_, cur.descriptors, n_)
            : lighterglue_->run(prev_kpts_, prev_desc_, cur.keypoints,
                                cur.descriptors);
    for (int i = 0; i < n_; ++i) {
      if (prev_id_[i] < 0) continue;
      int j = m.matches0[i];
      if (j < 0 || j >= n_) continue;
      if (m.mscores0[i] < params.xfeat_min_conf) continue;
      if (cur_id[j] < 0 || m.mscores0[i] > best[j]) {
        cur_id[j] = prev_id_[i];
        cur_cnt[j] = prev_track_cnt_[i] + 1;
        best[j] = m.mscores0[i];
      }
    }
  }

  double t_match = t_step.toc();
  t_step.tic();

  int n_matched = 0;  // tracks that re-matched this frame (continuation)
  for (int j = 0; j < n_; ++j)
    if (cur_id[j] >= 0) n_matched++;

  // 3. Select a spatially-spread subset up to max_cnt: tracked first (longest
  //    tracks), then new high-score keypoints (mirrors KLT setMask).
  cv::Mat mask(row_, col_, CV_8UC1, cv::Scalar(255));
  std::vector<int> sel;
  sel.reserve(params.max_cnt);
  const int min_dist = params.min_dist;

  auto try_accept = [&](int slot) {
    if (static_cast<int>(sel.size()) >= params.max_cnt) return false;
    const cv::Point2f &p = cur.keypoints[slot];
    if (!inBorder(p)) return false;
    if (mask.at<uchar>(cvRound(p.y), cvRound(p.x)) != 255) return false;
    sel.push_back(slot);
    cv::circle(mask, p, min_dist, cv::Scalar(0), -1);
    return true;
  };

  // Pass 1: tracked slots, longest tracks first.
  std::vector<int> tracked;
  for (int j = 0; j < n_; ++j)
    if (cur_id[j] >= 0) tracked.push_back(j);
  std::sort(tracked.begin(), tracked.end(),
            [&](int a, int b) { return cur_cnt[a] > cur_cnt[b]; });
  for (int slot : tracked) try_accept(slot);

  // Pass 2: new keypoints above score threshold, highest score first.
  std::vector<int> fresh;
  for (int j = 0; j < n_; ++j)
    if (cur_id[j] < 0 && cur.scores[j] >= params.xfeat_score_thr) fresh.push_back(j);
  std::sort(fresh.begin(), fresh.end(),
            [&](int a, int b) { return cur.scores[a] > cur.scores[b]; });
  for (int slot : fresh) {
    if (try_accept(slot)) {
      cur_id[slot] = IdCounter::get();
      cur_cnt[slot] = 1;
    }
  }

  // 4. Build per-feature outputs for the selected slots.
  std::vector<cv::Point2f> cur_pts;
  std::vector<int> ids;
  std::vector<int> track_cnt;
  cur_pts.reserve(sel.size());
  ids.reserve(sel.size());
  track_cnt.reserve(sel.size());
  for (int slot : sel) {
    cur_pts.push_back(cur.keypoints[slot]);
    ids.push_back(cur_id[slot]);
    track_cnt.push_back(cur_cnt[slot]);
  }

  std::vector<cv::Point2f> un_pts = undistortedPts(cur_pts, m_camera_[0]);
  std::map<int, cv::Point2f> cur_un_pts_map;
  std::vector<cv::Point2f> velocity =
      ptsVelocity(ids, un_pts, cur_un_pts_map, prev_un_pts_map_,
                  cur_time - prev_time_);

  if (params.show_track) drawTrack(img, ids, cur_pts, track_cnt, prev_pts_map_);

  // 5. Persist current frame as the new "previous" (full N for matching).
  prev_kpts_ = cur.keypoints;
  prev_desc_ = cur.descriptors;
  prev_id_.assign(n_, -1);
  prev_track_cnt_.assign(n_, 0);
  for (int slot : sel) {
    prev_id_[slot] = cur_id[slot];
    prev_track_cnt_[slot] = cur_cnt[slot];
  }
  prev_un_pts_map_ = cur_un_pts_map;
  prev_pts_map_.clear();
  for (size_t i = 0; i < ids.size(); ++i) prev_pts_map_[ids[i]] = cur_pts[i];
  prev_time_ = cur_time;
  has_prev_ = true;

  // 6. Assemble the canonical VINS feature frame (mono => camera_id 0).
  std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 8, 1>>>> frame;
  for (size_t i = 0; i < ids.size(); ++i) {
    Eigen::Matrix<double, 8, 1> xyz_uv_velocity;
    xyz_uv_velocity << un_pts[i].x, un_pts[i].y, 1.0, cur_pts[i].x, cur_pts[i].y,
        velocity[i].x, velocity[i].y;
    frame[ids[i]].emplace_back(0, xyz_uv_velocity);
  }
  int n_tracked_sel = 0;
  for (int c : track_cnt)
    if (c >= 2) n_tracked_sel++;
  ROS_INFO(
      "XFeat track: %zu feat | prevTrk %d matched %d (%.0f%%) keptTrk %d evicted %d "
      "| extract %.1f match %.1f rest %.1f total %.1f ms",
      ids.size(), n_prev_tracks, n_matched,
      n_prev_tracks ? 100.0 * n_matched / n_prev_tracks : 0.0, n_tracked_sel,
      n_matched - n_tracked_sel, t_extract, t_match, t_step.toc(), t_all.toc());
  return frame;
}

void FeatureTrackerXFeat::removeOutliers(std::set<int> &removePtsIds) {
  for (int i = 0; i < n_; ++i) {
    if (prev_id_[i] >= 0 && removePtsIds.count(prev_id_[i])) {
      prev_id_[i] = -1;
      prev_track_cnt_[i] = 0;
    }
  }
}

cv::Mat FeatureTrackerXFeat::getTrackImage() { return im_track_; }

}  // namespace vins::estimator
