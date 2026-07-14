/*******************************************************
 * node_gana.cpp
 *
 * GANA (Gorsel Ataletsel Navigasyon Algoritmasi = Visual-Inertial Navigation
 * Algorithm) endpoint for the MORS / NOVA / ASPN messaging scheme.
 *
 * Responsibilities:
 *   1. Serve  /gana/alg_control  (mors_msgs/AlgControl): START/STOP + initial_pva.
 *   2. Subscribe to loop-closure-corrected VIO pose  /loop_fusion/odometry_rect
 *      and to raw odometry  /vins_estimator/odometry  (for velocity).
 *   3. Convert VINS world-frame PVA -> ASPN GEODETIC PVA.
 *   4. Maintain a sliding-window trajectory and publish
 *      nova_msgs/MeasurementPositionVelocityAttitudeTrajectory on  /gana .
 *
 * Frame design (Design B): node_gana OWNS the local->global alignment, captured
 * once at START from the master's initial_pva. No GPS, no global_fusion. Loop
 * closure stays in loop_fusion (local) and is consumed via odometry_rect.
 *
 *   W   = VINS world frame (gravity-aligned z-up, arbitrary yaw)
 *   NED = local-level North-East-Down at the geodetic origin (initial_pva)
 *   b   = body/sensor frame
 *
 * Captured at START:
 *   geo origin   = (lat0, lon0, alt0)  from initial_pva
 *   R_ned_W      = R_ned_b0 * R_W_b0^T   (absorbs the unknown VIO yaw)
 *   P_W0         = body position in W at START
 * Per corrected odom:
 *   P_ned = R_ned_W * (P_W - P_W0);  ENU=(E,N,U)=(P_ned.y,P_ned.x,-P_ned.z)
 *   lat/lon/alt = geo.Reverse(E,N,U)
 *   V_ned = R_ned_W * V_W
 *   R_ned_b = R_ned_W * R_W_b   -> ASPN quaternion [w,x,y,z]
 *******************************************************/

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>

#include <aspn_msgs/MeasurementPositionVelocityAttitude.h>
#include <nova_msgs/MeasurementPositionVelocityAttitudeTrajectory.h>
#include <mors_msgs/AlgControl.h>

#include <GeographicLib/LocalCartesian.hpp>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <deque>
#include <map>
#include <mutex>
#include <array>
#include <cmath>

namespace {
constexpr double RAD2DEG = 180.0 / M_PI;
constexpr double DEG2RAD = M_PI / 180.0;

// ---- ASPN quaternion convention --------------------------------------------
// ASPN stores attitude as the rotation applied to the reference (NED) frame to
// bring it onto the body axes, ordered [w, x, y, z]. We treat q.toMatrix() as
// R_b_ned (body <- ned). Both helpers live here so the convention is defined in
// ONE place; if a known-heading bag shows the heading is mirrored, flip here.
// TODO(validate): confirm sign/handedness against a bag with known true heading.
Eigen::Matrix3d aspnQuatToR_b_ned(const std::array<double, 4> &q /*w,x,y,z*/) {
  Eigen::Quaterniond eq(q[0], q[1], q[2], q[3]);  // (w,x,y,z)
  eq.normalize();
  return eq.toRotationMatrix();
}
std::array<double, 4> rToAspnQuat_b_ned(const Eigen::Matrix3d &R_b_ned) {
  Eigen::Quaterniond eq(R_b_ned);
  eq.normalize();
  return {eq.w(), eq.x(), eq.y(), eq.z()};
}
}  // namespace

class GanaNode {
 public:
  explicit GanaNode(ros::NodeHandle &nh, ros::NodeHandle &pnh) {
    pnh.param<std::string>("pose_topic", pose_topic_, "/loop_fusion/odometry_rect");
    pnh.param<std::string>("vel_topic", vel_topic_, "/vins_estimator/odometry");
    pnh.param<std::string>("gana_topic", gana_topic_, "/gana");
    pnh.param<std::string>("alg_control_service", srv_name_, "/gana/alg_control");
    pnh.param<double>("trajectory_history_duration_sec", history_sec_, 5.0);
    // Placeholder 1-sigma uncertainties until the real VINS marginal is wired in.
    pnh.param<double>("pos_std_m", pos_std_, 0.5);
    pnh.param<double>("vel_std_mps", vel_std_, 0.1);
    pnh.param<double>("att_std_rad", att_std_, 0.02);

    pub_ = nh.advertise<nova_msgs::MeasurementPositionVelocityAttitudeTrajectory>(
        gana_topic_, 10);
    sub_pose_ = nh.subscribe(pose_topic_, 200, &GanaNode::poseCallback, this);
    sub_vel_ = nh.subscribe(vel_topic_, 200, &GanaNode::velCallback, this);
    srv_ = nh.advertiseService(srv_name_, &GanaNode::handleAlgControl, this);

    ROS_INFO("GANA node up. Waiting for START on %s", srv_name_.c_str());
    ROS_INFO("  pose(in): %s   vel(in): %s   traj(out): %s",
             pose_topic_.c_str(), vel_topic_.c_str(), gana_topic_.c_str());
  }

 private:
  // ----- AlgControl service -------------------------------------------------
  bool handleAlgControl(mors_msgs::AlgControl::Request &req,
                        mors_msgs::AlgControl::Response &res) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (req.command == mors_msgs::AlgControl::Request::START) {
      if (started_) {
        res.success = true;
        res.message = "Algorithm is already running.";
        return true;
      }
      if (req.initial_pva.reference_frame !=
          aspn_msgs::MeasurementPositionVelocityAttitude::GEODETIC) {
        res.success = false;
        res.message = "invalid initial_pva: reference_frame must be GEODETIC.";
        ROS_ERROR("%s", res.message.c_str());
        return true;  // service call itself succeeded; result reports failure
      }
      initial_pva_ = req.initial_pva;
      // Set the geodetic origin (GeographicLib uses DEGREES; ASPN p1/p2 are rad).
      geo_.Reset(initial_pva_.p1 * RAD2DEG, initial_pva_.p2 * RAD2DEG,
                 initial_pva_.p3);
      alignment_ready_ = false;   // captured on the first corrected odom
      buffer_.clear();
      vel_cache_.clear();
      started_ = true;
      res.success = true;
      res.message = "Algorithm started.";
      ROS_INFO("START: origin lat=%.7f lon=%.7f alt=%.2f",
               initial_pva_.p1 * RAD2DEG, initial_pva_.p2 * RAD2DEG,
               initial_pva_.p3);
      return true;
    }
    if (req.command == mors_msgs::AlgControl::Request::STOP) {
      res.success = true;
      res.message = started_ ? "Algorithm stopped." : "Algorithm already stopped.";
      started_ = false;
      ROS_INFO("STOP");
      return true;
    }
    res.success = false;
    res.message = "Invalid algorithm control command.";
    ROS_ERROR("%s", res.message.c_str());
    return true;
  }

  // ----- velocity cache (raw odom carries twist; odometry_rect does not) -----
  void velCallback(const nav_msgs::Odometry::ConstPtr &msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    const int64_t t = msg->header.stamp.toNSec();
    vel_cache_[t] = Eigen::Vector3d(msg->twist.twist.linear.x,
                                    msg->twist.twist.linear.y,
                                    msg->twist.twist.linear.z);
    pruneVelCache(t);
  }

  // ----- main path: corrected pose -> ASPN PVA ------------------------------
  void poseCallback(const nav_msgs::Odometry::ConstPtr &msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!started_) return;

    const Eigen::Vector3d P_W(msg->pose.pose.position.x,
                              msg->pose.pose.position.y,
                              msg->pose.pose.position.z);
    const Eigen::Quaterniond Q_W(msg->pose.pose.orientation.w,
                                 msg->pose.pose.orientation.x,
                                 msg->pose.pose.orientation.y,
                                 msg->pose.pose.orientation.z);
    const Eigen::Matrix3d R_W_b = Q_W.normalized().toRotationMatrix();

    if (!alignment_ready_) {
      // R_ned_W = R_ned_b0 * R_W_b0^T, with R_ned_b0 = (R_b_ned0)^T.
      const Eigen::Matrix3d R_b_ned0 = aspnQuatToR_b_ned(
          {initial_pva_.quaternion[0], initial_pva_.quaternion[1],
           initial_pva_.quaternion[2], initial_pva_.quaternion[3]});
      R_ned_W_ = R_b_ned0.transpose() * R_W_b.transpose();
      P_W0_ = P_W;
      alignment_ready_ = true;
      ROS_INFO("Captured W->NED alignment from initial_pva.");
    }

    // Position: W -> NED -> ENU -> geodetic.
    const Eigen::Vector3d P_ned = R_ned_W_ * (P_W - P_W0_);
    double lat_deg, lon_deg, alt_m;
    geo_.Reverse(/*E=*/P_ned.y(), /*N=*/P_ned.x(), /*U=*/-P_ned.z(),
                 lat_deg, lon_deg, alt_m);

    // Velocity: match raw-odom twist by timestamp (same header as the rect msg).
    Eigen::Vector3d V_ned = Eigen::Vector3d::Zero();
    if (lookupVelocity(msg->header.stamp.toNSec(), &V_ned)) {
      V_ned = R_ned_W_ * V_ned;  // V_W -> NED
    } else {
      ROS_WARN_THROTTLE(2.0, "No matching velocity for pose stamp; using 0.");
    }

    // Attitude: R_ned_b = R_ned_W * R_W_b  ->  emit as R_b_ned (ASPN convention).
    const Eigen::Matrix3d R_ned_b = R_ned_W_ * R_W_b;
    const std::array<double, 4> q_aspn = rToAspnQuat_b_ned(R_ned_b.transpose());

    aspn_msgs::MeasurementPositionVelocityAttitude m;
    m.time_of_validity.elapsed_nsec = msg->header.stamp.toNSec();
    m.reference_frame = aspn_msgs::MeasurementPositionVelocityAttitude::GEODETIC;
    m.p1 = lat_deg * DEG2RAD;
    m.p2 = lon_deg * DEG2RAD;
    m.p3 = alt_m;
    m.v1 = V_ned.x();
    m.v2 = V_ned.y();
    m.v3 = V_ned.z();
    m.quaternion = {q_aspn[0], q_aspn[1], q_aspn[2], q_aspn[3]};
    m.num_meas = 9;
    m.covariance = diagCovariance();  // TODO: real VINS marginal
    m.error_model = aspn_msgs::MeasurementPositionVelocityAttitude::NONE;
    m.num_error_model_params = 0;
    m.num_integrity = 0;

    // Sliding window: overwrite same-timestamp entry (loop-closure re-publish),
    // keep time-ordered, prune beyond history window.
    upsert(m);
    publishTrajectory(msg->header.stamp.toNSec());
  }

  bool lookupVelocity(int64_t t, Eigen::Vector3d *out) {
    auto it = vel_cache_.find(t);
    if (it == vel_cache_.end()) {
      if (vel_cache_.empty()) return false;
      it = std::prev(vel_cache_.end());  // fall back to most recent
    }
    *out = it->second;
    return true;
  }

  void upsert(const aspn_msgs::MeasurementPositionVelocityAttitude &m) {
    const int64_t t = m.time_of_validity.elapsed_nsec;
    for (auto &e : buffer_) {
      if (e.time_of_validity.elapsed_nsec == t) { e = m; return; }
    }
    buffer_.push_back(m);
    const int64_t cutoff = t - static_cast<int64_t>(history_sec_ * 1e9);
    while (!buffer_.empty() &&
           buffer_.front().time_of_validity.elapsed_nsec < cutoff) {
      buffer_.pop_front();
    }
  }

  void pruneVelCache(int64_t now_ns) {
    const int64_t cutoff = now_ns - static_cast<int64_t>(history_sec_ * 1e9);
    while (!vel_cache_.empty() && vel_cache_.begin()->first < cutoff) {
      vel_cache_.erase(vel_cache_.begin());
    }
  }

  void publishTrajectory(int64_t gen_ns) {
    nova_msgs::MeasurementPositionVelocityAttitudeTrajectory traj;
    traj.time_of_generation.elapsed_nsec = gen_ns;
    traj.trajectory_history_duration_sec = history_sec_;
    traj.num_measurements = static_cast<uint32_t>(buffer_.size());
    traj.measurements.assign(buffer_.begin(), buffer_.end());
    pub_.publish(traj);
  }

  // Diagonal 9x9 (row-major) covariance: index i*9+i. Order [N,E,D,vN,vE,vD,
  // phiN,phiE,phiD]. (NOTE: the ROS2 demo packed off-diagonal indices; that was
  // a bug. The diagonal stride is 10, not 9.)
  std::vector<double> diagCovariance() const {
    std::vector<double> c(81, 0.0);
    const double pv = pos_std_ * pos_std_;
    const double vv = vel_std_ * vel_std_;
    const double av = att_std_ * att_std_;
    for (int i = 0; i < 3; ++i) c[(0 + i) * 9 + (0 + i)] = pv;
    for (int i = 0; i < 3; ++i) c[(3 + i) * 9 + (3 + i)] = vv;
    for (int i = 0; i < 3; ++i) c[(6 + i) * 9 + (6 + i)] = av;
    return c;
  }

  // config
  std::string pose_topic_, vel_topic_, gana_topic_, srv_name_;
  double history_sec_ = 5.0, pos_std_ = 0.5, vel_std_ = 0.1, att_std_ = 0.02;

  // ros
  ros::Publisher pub_;
  ros::Subscriber sub_pose_, sub_vel_;
  ros::ServiceServer srv_;

  // state (guarded by mtx_)
  std::mutex mtx_;
  bool started_ = false;
  bool alignment_ready_ = false;
  aspn_msgs::MeasurementPositionVelocityAttitude initial_pva_;
  GeographicLib::LocalCartesian geo_;
  Eigen::Matrix3d R_ned_W_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d P_W0_ = Eigen::Vector3d::Zero();
  std::deque<aspn_msgs::MeasurementPositionVelocityAttitude> buffer_;
  std::map<int64_t, Eigen::Vector3d> vel_cache_;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "gana_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  GanaNode node(nh, pnh);
  ros::spin();
  return 0;
}
