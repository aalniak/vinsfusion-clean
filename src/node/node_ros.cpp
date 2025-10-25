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

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <stdio.h>
#include <vins_estimator/estimator/estimator.h>
#include <vins_estimator/estimator/parameters.h>
#include <vins_estimator/utility/visualization.h>

#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <functional>
#include "vins/EstimateDepth.h"

using namespace vins::estimator;

std::unique_ptr<Estimator> estimator;
std::unique_ptr<Parameters> params;

queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg) {
  m_buf.lock();
  img0_buf.push(img_msg);
  m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg) {
  m_buf.lock();
  img1_buf.push(img_msg);
  m_buf.unlock();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
  cv_bridge::CvImageConstPtr ptr;
  if (img_msg->encoding == "8UC1") {
    sensor_msgs::Image img;
    img.header = img_msg->header;
    img.height = img_msg->height;
    img.width = img_msg->width;
    img.is_bigendian = img_msg->is_bigendian;
    img.step = img_msg->step;
    img.data = img_msg->data;
    img.encoding = "mono8";
    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  } else
    ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

  cv::Mat img = ptr->image.clone();
  return img;
}

// extract images with same timestamp from two topics
void sync_process(ros::ServiceClient& depth_client) {
  while (ros::ok()) {
    if (params->stereo) {
      cv::Mat image0;
      cv::Mat image1;
      std_msgs::Header header;
      double time = 0;
      m_buf.lock();
      if (!img0_buf.empty() && !img1_buf.empty()) {
        double time0 = img0_buf.front()->header.stamp.toSec();
        double time1 = img1_buf.front()->header.stamp.toSec();
        // 0.003s sync tolerance
        if (time0 < time1 - 0.003) {
          img0_buf.pop();
          printf("throw img0\n");
        } else if (time0 > time1 + 0.003) {
          img1_buf.pop();
          printf("throw img1\n");
        } else {
          time = img0_buf.front()->header.stamp.toSec();
          header = img0_buf.front()->header;
          image0 = getImageFromMsg(img0_buf.front());
          img0_buf.pop();
          image1 = getImageFromMsg(img1_buf.front());
          img1_buf.pop();
          // printf("find img0 and img1\n");
        }
      }
      m_buf.unlock();
      if (!image0.empty()) estimator->inputImage(time, image0, cv::Mat(), image1); // no depth support for stereo case
    } else { //add depth stuff only into monocular case
      cv::Mat image;
      std_msgs::Header header;
      double time = 0;
      sensor_msgs::ImageConstPtr img_msg_ptr;
      m_buf.lock();
      if (!img0_buf.empty()) {
        time = img0_buf.front()->header.stamp.toSec();
        header = img0_buf.front()->header;
        img_msg_ptr = img0_buf.front();
        image = getImageFromMsg(img0_buf.front());
        img0_buf.pop();
      }
      m_buf.unlock();
      if (!image.empty()) {
          cv::Mat depth_image;
          vins::EstimateDepth srv;
          srv.request.input_image = *img_msg_ptr; //original message goes in

          if (depth_client.call(srv)) {
              // We got a response
              ROS_DEBUG("Successfully received depth map!");
              try {
                  // Convert the 32FC1 ROS message back to an OpenCV Mat
                  cv_bridge::CvImageConstPtr cv_ptr;
                  cv_ptr = cv_bridge::toCvCopy(srv.response.depth_map, sensor_msgs::image_encodings::TYPE_32FC1);
                  depth_image = cv_ptr->image;
              } catch (cv_bridge::Exception& e) {
                  ROS_ERROR("cv_bridge exception: %s", e.what());
              }
          } else {
              ROS_ERROR("Failed to call depth service. Is the python node running?");
              // Decide how to handle failure: continue, return, or send empty Mat
          }
          estimator->inputImage(time, image, depth_image);
      }
    }

    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg) {
  double t = imu_msg->header.stamp.toSec();
  double dx = imu_msg->linear_acceleration.x;
  double dy = imu_msg->linear_acceleration.y;
  double dz = imu_msg->linear_acceleration.z;
  double rx = imu_msg->angular_velocity.x;
  double ry = imu_msg->angular_velocity.y;
  double rz = imu_msg->angular_velocity.z;
  Vector3d acc(dx, dy, dz);
  Vector3d gyr(rx, ry, rz);
  estimator->inputIMU(t, acc, gyr);
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg) {
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
  for (unsigned i = 0; i < feature_msg->points.size(); i++) {
    int feature_id = feature_msg->channels[0].values[i];
    int camera_id = feature_msg->channels[1].values[i];
    double x = feature_msg->points[i].x;
    double y = feature_msg->points[i].y;
    double z = feature_msg->points[i].z;
    double p_u = feature_msg->channels[2].values[i];
    double p_v = feature_msg->channels[3].values[i];
    double velocity_x = feature_msg->channels[4].values[i];
    double velocity_y = feature_msg->channels[5].values[i];
    // if (feature_msg->channels.size() > 5) {
    //   double gx = feature_msg->channels[6].values[i];
    //   double gy = feature_msg->channels[7].values[i];
    //   double gz = feature_msg->channels[8].values[i];
    //   // pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
    //   // printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
    // }
    ROS_ASSERT(z == 1);
    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
    featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
  }
  double t = feature_msg->header.stamp.toSec();
  estimator->inputFeature(t, featureFrame);
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg) {
  if (restart_msg->data == 1) {
    ROS_WARN("restart the estimator!");
    estimator->clearState();
    estimator->setParameter();
  }
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg) {
  if (switch_msg->data == 1) {
    // ROS_WARN("use IMU!");
    estimator->changeSensorType(1, params->stereo);
  } else {
    // ROS_WARN("disable IMU!");
    estimator->changeSensorType(0, params->stereo);
  }
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg) {
  if (switch_msg->data == 1) {
    // ROS_WARN("use stereo!");
    estimator->changeSensorType(params->use_imu, 1);
  } else {
    // ROS_WARN("use mono camera (left)!");
    estimator->changeSensorType(params->use_imu, 0);
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "vins_estimator");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Info);

  string config_file;
  if (n.getParam("config_path", config_file)) {
    ROS_INFO_STREAM("Successfully loaded config_file: " << config_file);
  } else {
    ROS_ERROR_STREAM("Failed to load config_file parameter.");
    return -1;
  }
  std::cout << "config_file: " << config_file << std::endl;

  // Initialize estimator and parameters
  params.reset(new Parameters());
  params->read_from_file(config_file);
  estimator.reset(new Estimator(*params));

  estimator->setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
  ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

  ROS_WARN("waiting for image and imu...");

  registerPub(n);

  ros::ServiceClient depth_client;
  depth_client = n.serviceClient<vins::EstimateDepth>("/estimate_depth");

  ROS_INFO("Waiting for depth service '/estimate_depth'...");
  depth_client.waitForExistence();
  ROS_INFO("Depth service connected!");

  ros::Subscriber sub_imu;
  if (params->use_imu) {
    sub_imu = n.subscribe(params->imu_topic, 2000, imu_callback,
                          ros::TransportHints().tcpNoDelay());
  }
  ros::Subscriber sub_feature =
      n.subscribe("/feature_tracker/feature", 2000, feature_callback);
  ros::Subscriber sub_img0 =
      n.subscribe(params->image0_topic, 100, img0_callback);
  ros::Subscriber sub_img1;
  if (params->stereo) {
    sub_img1 = n.subscribe(params->image1_topic, 100, img1_callback);
  }
  ros::Subscriber sub_restart =
      n.subscribe("/vins_restart", 100, restart_callback);
  ros::Subscriber sub_imu_switch =
      n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
  ros::Subscriber sub_cam_switch =
      n.subscribe("/vins_cam_switch", 100, cam_switch_callback);

  std::thread sync_thread{sync_process, std::ref(depth_client)};
  ros::spin();

  if (sync_thread.joinable()) {
    sync_thread.join();
  }

  return 0;
}
