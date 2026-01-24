/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vins_estimator/estimator/estimator.h>
#include <vins_estimator/estimator/parameters.h>
#include <vins_estimator/utility/visualization.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp> // For debug purposes.

#include <cassert>
#include <cstddef>
#include <fstream>
#include <algorithm>  // For std::sort, std::nth_element
#include <vins_estimator/factor/ordinalDepthFactor.h>  // For OrdinalDepthFactor

namespace vins::estimator {
// ------- Depth factor declaration ------- 
// This factor pulls VIO inverse depth toward the aligned monocular depth prediction
// Using a GLOBAL (temporally smoothed) scale and shift, not per-frame optimization
struct DepthPriorFactor
{
    const double aligned_inv_depth;  // = global_scale * mono_inv_depth + global_shift
    const double sqrt_info;
    const bool use_log;              // true: log-domain residual, false: linear residual

    DepthPriorFactor(double aligned_inv_d, double weight, bool log_residual = false)
        : aligned_inv_depth(aligned_inv_d), sqrt_info(weight), use_log(log_residual) {}

    template <typename T>
    bool operator()(const T* const inv_depth_vio, T* residuals) const
    {
        if (use_log) {
            // Log-domain residual: both depths are in inverse domain
            // Converting to log space makes the residual scale-invariant and more robust
            // log(inv_d_vio) - log(inv_d_prior) = log(inv_d_vio / inv_d_prior)
            // This is equivalent to relative depth error in log space
            residuals[0] = T(sqrt_info) * (ceres::log(inv_depth_vio[0]) - ceres::log(T(aligned_inv_depth)));
        } else {
            // Linear residual in inverse depth domain
            residuals[0] = T(sqrt_info) * (inv_depth_vio[0] - T(aligned_inv_depth));
        }
        return true;
    }

    static ceres::CostFunction* Create(double aligned_inv_d, double weight, bool log_residual = false) {
        return new ceres::AutoDiffCostFunction<DepthPriorFactor, 1, 1>(
            new DepthPriorFactor(aligned_inv_d, weight, log_residual));
    }
};

// Legacy factor with per-frame scale/shift (kept for reference, but not recommended)
struct VioDisparityModelFactor
{
    // The measurement from the network is a standard metric DEPTH value.
    const double depth_network;
    const double sqrt_info;

    VioDisparityModelFactor(double depth_net, double weight)
        : depth_network(depth_net), sqrt_info(weight) {}

    template <typename T>
    bool operator()(const T* const inv_depth_vio,      
                    const T* const scale_shift_prime,   
                    T* residuals) const
    {
        if (depth_network <= 1e-6) {
            residuals[0] = T(0.0);
            return true;
        }

        const T rho_net = T(depth_network); // inv depth from network

        const T& rho_vio = inv_depth_vio[0]; // inv depth from vins


        
        const T& s_prime = scale_shift_prime[0]; // scale on rho
        const T& b_prime = scale_shift_prime[1]; // shift on rho

        T rho_net_corrected = s_prime * rho_net + b_prime;

        residuals[0] = T(sqrt_info) * (rho_vio - rho_net_corrected);
        //std::cout << "The metric residual for this very feature with vins depth "<< (T(1.0) / rho_vio) << " and aligned depth " << (T(1.0) / rho_net_corrected) << "is: " << (T(1.0) / rho_vio) - (T(1.0) / rho_net_corrected) << std::endl;
        return true;
    }

    // The create function takes depth_net and instantiates the factor.
    static ceres::CostFunction* Create(double depth_net, double weight) {
        return (new ceres::AutoDiffCostFunction<VioDisparityModelFactor, 1, 1, 2>(
            new VioDisparityModelFactor(depth_net, weight)));
    }
};
// ----------------------------------------------- 
// For Debug: Save image to folder
void saveImageToFolder(const cv::Mat& image, const std::string& folderPath, const std::string& filename) {
    
    // 1. Check if the image is valid
    if (image.empty()) {
        std::cerr << "Error: Image is empty!" << std::endl;
        return;
    }

    // 2. Construct the full path
    // Ensure the folder path ends with a slash (Linux/Mac use '/', Windows uses '\\')
    std::string fullPath;
    if (folderPath.back() == '/' || folderPath.back() == '\\') {
        fullPath = folderPath + filename;
    } else {
        fullPath = folderPath + "/" + filename;
    }

    // 3. Save the image
    // Returns true on success, false on failure (e.g., if folder doesn't exist)
    bool success = cv::imwrite(fullPath, image);

    if (success) {
        std::cout << "Image saved successfully to: " << fullPath << std::endl;
    } else {
        std::cerr << "Error: Failed to save image. Does the folder '" << folderPath << "' exist?" << std::endl;
    }
}

inline float getBilinearDepth(const cv::Mat& depth_map, float x, float y) {
    // Boundary check
    if (x < 0 || x >= depth_map.cols - 1 || y < 0 || y >= depth_map.rows - 1)
        return -1.0f;

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float dx = x - x0;
    float dy = y - y0;

    // Fast pointer access to rows (Avoids .at() overhead)
    const float* row0 = depth_map.ptr<float>(y0);
    const float* row1 = depth_map.ptr<float>(y1);

    float val00 = row0[x0];
    float val10 = row0[x1];
    float val01 = row1[x0];
    float val11 = row1[x1];

    // Bilinear interpolation
    float top = val00 * (1.0f - dx) + val10 * dx;
    float bot = val01 * (1.0f - dx) + val11 * dx;
    return top * (1.0f - dy) + bot * dy;
}


Estimator::Estimator(Parameters &params)
    : f_manager{params},
      featureTracker{params},
      params{params},
      initial_ex_rotation{params} {
  ROS_INFO("init begins");
  clearState();
  std::string engine_path = params.depth_engine_path;
  depthInferer = std::make_shared<DepthInfer>(engine_path);
  std::cout << "Engine is loaded" << std::endl;
  // Global or member variable in Estimator class
  //SplgInference* sp_lg_inferer = new SplgInference("/datasets/splg_1280x800_fp16.engine");
}

Estimator::~Estimator() {
  if (params.multiple_thread) {
    if (processThread.joinable()) {
      processThread.join();
    }
    if (trackThread.joinable()) {
      trackThread.join();
    }
    printf("join thread \n");
  }
}

void Estimator::clearState() {
  mProcess.lock();
  while (!accBuf.empty()) accBuf.pop();
  while (!gyrBuf.empty()) gyrBuf.pop();
  while (!featureBuf.empty()) featureBuf.pop();
  
  // ------- Depth scale&shift variable allocation ------- 
  for (int i = 0; i < WINDOW_SIZE + 1; i++)
  {
    depth_net_scales[i] = 1.0;
    depth_net_shifts[i] = 0.0;
  }
  //  -----------------------------------------------------
  
  prevTime = -1;
  curTime = 0;
  openExEstimation = false;
  initP = Eigen::Vector3d(0, 0, 0);
  initR = Eigen::Matrix3d::Identity();
  inputImageCnt = 0;
  initFirstPoseFlag = false;

  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    Rs[i].setIdentity();
    Ps[i].setZero();
    Vs[i].setZero();
    Bas[i].setZero();
    Bgs[i].setZero();
    dt_buf[i].clear();
    linear_acceleration_buf[i].clear();
    angular_velocity_buf[i].clear();

    if (pre_integrations[i] != nullptr) {
      delete pre_integrations[i];
    }
    pre_integrations[i] = nullptr;
  }

  for (int i = 0; i < params.num_of_cam; i++) {
    tic[i] = Vector3d::Zero();
    ric[i] = Matrix3d::Identity();
  }

  first_imu = false, sum_of_back = 0;
  sum_of_front = 0;
  frame_count = 0;
  solver_flag = INITIAL;
  initial_timestamp = 0;
  all_image_frame.clear();

  delete tmp_pre_integration;
  delete last_marginalization_info;

  tmp_pre_integration = nullptr;
  last_marginalization_info = nullptr;
  last_marginalization_parameter_blocks.clear();

  f_manager.clearState();

  failure_occur = false;

  mProcess.unlock();
}

void Estimator::setParameter() {
  mProcess.lock();
  for (int i = 0; i < params.num_of_cam; i++) {
    tic[i] = params.tic[i];
    ric[i] = params.ric[i];
    cout << " exitrinsic cam " << i << endl
         << ric[i] << endl
         << tic[i].transpose() << endl;
  }
  ros::NodeHandle nh("~");
  nh.param("weight", WEIGHT, WEIGHT);
  f_manager.setRic(ric);
  ProjectionTwoFrameOneCamFactor::sqrt_info =
      params.focal_length / 1.5 * Matrix2d::Identity();
  ProjectionTwoFrameTwoCamFactor::sqrt_info =
      params.focal_length / 1.5 * Matrix2d::Identity();
  ProjectionOneFrameTwoCamFactor::sqrt_info =
      params.focal_length / 1.5 * Matrix2d::Identity();
  td = params.td;
  g = params.g;
  cout << "set g " << g.transpose() << endl;
  featureTracker.readIntrinsicParameter(params.cam_names);

  std::cout << "params.multiple_thread is " << params.multiple_thread << '\n';
  if (params.multiple_thread && !initThreadFlag) {
    initThreadFlag = true;
    processThread = std::thread(&Estimator::processMeasurements, this);
  }
  mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo) {
  bool restart = false;
  mProcess.lock();
  if (!use_imu && !use_stereo)
    printf("at least use two sensors! \n");
  else {
    if (params.use_imu != use_imu) {
      params.use_imu = use_imu;
      if (params.use_imu) {
        // reuse imu; restart system
        restart = true;
      } else {
        delete last_marginalization_info;

        tmp_pre_integration = nullptr;
        last_marginalization_info = nullptr;
        last_marginalization_parameter_blocks.clear();
      }
    }

    params.stereo = use_stereo;
    printf("use imu %d use stereo %d\n", params.use_imu, params.stereo);
  }
  mProcess.unlock();
  if (restart) {
    clearState();
    setParameter();
  }
}

void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &depth_img, 
                           const cv::Mat &_img1) {
  inputImageCnt++;
  std::cout << "Processing frame number " << inputImageCnt << std::endl;
  if (solver_flag == NON_LINEAR && nonlinear_input_cnt < 15)  {
    nonlinear_input_cnt++;
  }
  // 1. Prepare RGB image for Cache (and Inference)
  cv::Mat rgb_img;
  if (_img.channels() == 1) {
      cv::cvtColor(_img, rgb_img, cv::COLOR_GRAY2RGB);
  } else {
      rgb_img = _img.clone();
  }

  // 2. Cache Logic
  mCache.lock();
  image_cache[t] = rgb_img;
  frame_index_cache[t] = inputImageCnt - 1;  // Store absolute frame index (0-indexed)
  for(auto it = image_cache.begin(); it != image_cache.end(); ) {
      if(it->first < t - 4.0) {
          frame_index_cache.erase(it->first);  // Also clean up frame index cache
          it = image_cache.erase(it);
      }
      else ++it;
  }
  mCache.unlock();

  // 3. Inference & Depth Injection
  // Initialize with the original Mono8 image by default
  cv::Mat img_for_tracker = _img.clone(); 


  cv::Mat depth_8u;
 if (depthInferer && params.rgd && solver_flag != INITIAL){
        
        // --- TIMER: INFERENCE ---
        TicToc t_infer; 
        cv::Mat raw_inv_depth;
        if (!params.use_gt){
        // A. Run Inference (Small Image -> Small Float Map)
        raw_inv_depth = depthInferer->infer(rgb_img);
 

        
        }
        else {
          // Use ground truth .tiff file from params.depth_folder/XXXXXX_lcam_front_depth.tiff
          // Frame index is based on inputImageCnt (0-indexed)
          char frame_str[16];
          snprintf(frame_str, sizeof(frame_str), "%06d", inputImageCnt - 1);  // -1 because inputImageCnt is incremented at start
          std::string gt_depth_path = params.depth_folder + "/" + std::string(frame_str) + "_lcam_front_depth.tiff";
          
          cv::Mat gt_depth = cv::imread(gt_depth_path, cv::IMREAD_UNCHANGED);
          if (gt_depth.empty()) {
              std::cerr << "Error: Could not load ground truth depth image from " << gt_depth_path << std::endl;
              return;
          }
          else{
            std::cout << "Loaded ground truth depth image from " << gt_depth_path << std::endl;
          }
          // convert to inverse depth
          
          cv::divide(1.0, gt_depth, raw_inv_depth, 1.0, CV_32F);
        }
        double time_infer = t_infer.toc();
        // --- TIMER: POST-PROCESSING ---
        TicToc t_proc;
        
        // [OPTIMIZATION A]: Math on Small Float Image (518x518)
        // NOTE: We keep working in inverse depth space for numerical stability.
        // Inverse depth is bounded [0, inf) -> [inf, 0) in metric, which avoids:
        //   1. Division by near-zero depths
        //   2. Unbounded values for distant objects
        // The scale/shift fitting (s,t) is learned in inverse depth space:
        //   metric_inv_depth = s * mono_inv_depth + t
        //   metric_depth = 1.0 / metric_inv_depth (only when needed)
        cv::Mat inv_depth = raw_inv_depth;  // Renamed for clarity - this IS inverse depth

        // --- OPTIMIZATION: STRIDED SAMPLING (Approx 0.05ms) ---
        // We sample ~600 pixels to estimate the distribution.
        // This avoids iterating the whole 518x518 image.
        
        // Define sample size (hardcoded for speed, prevents allocation)
        // 518*518 / 431 is roughly 622 samples. 
        // Using a prime number stride prevents aliasing patterns.
        const int STRIDE = 101; 
        const int MAX_SAMPLES = 2700; 
        float samples[MAX_SAMPLES]; 
        
        int sample_count = 0;
        const float* ptr = (float*)inv_depth.data;
        const int total_pixels = inv_depth.rows * inv_depth.cols;

        // 1. FAST GATHER
        for (int i = 0; i < total_pixels && sample_count < MAX_SAMPLES; i += STRIDE) {
            samples[sample_count++] = ptr[i];
        }

        double robustMin, robustMax;

        if (sample_count > 10) {
            // 2. PARTIAL SORT (nth_element is O(N) on the small sample buffer)
            // Find lower 2.5%
            int idx_low = sample_count * 0.05;
            std::nth_element(samples, samples + idx_low, samples + sample_count);
            robustMin = samples[idx_low];

            // Find upper 97.5%
            int idx_high = sample_count * 0.95;
            // Note: We continue from the previous sort state
            std::nth_element(samples + idx_low + 1, samples + idx_high, samples + sample_count);
            robustMax = samples[idx_high];
        } else {
            // Fallback if image is tiny or something failed
            cv::minMaxLoc(inv_depth, &robustMin, &robustMax);
        }
        
        // Safety clamp to prevent div by zero
        if (robustMax <= robustMin + 1e-5) {
            robustMax = robustMin + 1.0;
        }

        cv::Mat depth_small_8u;
        const int num_pixels = inv_depth.rows * inv_depth.cols;

        if (params.metric_depth_vis == 1) {
            // --- METRIC DEPTH VISUALIZATION (Log-scaled, robust for indoor & outdoor) ---
            // Convert inverse depth to metric depth: depth = 1 / inv_depth
            
            cv::Mat metric_depth(inv_depth.size(), CV_32FC1);
            
            // Clamp inverse depth to reasonable range before converting
            // inv_depth ~0.005 -> depth ~200m (far outdoor)
            // inv_depth ~5.0   -> depth ~0.2m (very close)
            const float inv_depth_min_clamp = 0.005f;  // Max metric depth ~200m
            const float inv_depth_max_clamp = 5.0f;    // Min metric depth ~0.2m
            
            // Convert to metric depth with clamping
            const float* src_ptr = (const float*)inv_depth.data;
            float* dst_ptr = (float*)metric_depth.data;
            
            for (int i = 0; i < num_pixels; i++) {
                float inv_d = std::max(inv_depth_min_clamp, std::min(inv_depth_max_clamp, src_ptr[i]));
                dst_ptr[i] = 1.0f / inv_d;  // Now in meters
            }
            
            // --- LOGARITHMIC SCALING (Robust for indoor & outdoor) ---
            // log(depth) compresses the range: log(0.2m)=-1.6, log(1m)=0, log(10m)=2.3, log(200m)=5.3
            // This prevents outdoor far objects from dominating the visualization
            cv::Mat log_depth;
            cv::log(metric_depth + 0.1f, log_depth);  // +0.1 to avoid log(0)
            
            // Get robust min/max from log domain using sampled percentiles
            float log_samples[MAX_SAMPLES];
            const float* log_ptr = (const float*)log_depth.data;
            for (int i = 0; i < sample_count; i++) {
                int idx = i * STRIDE;
                if (idx < num_pixels) log_samples[i] = log_ptr[idx];
            }
            
            int log_idx_low = sample_count * 0.02;
            int log_idx_high = sample_count * 0.98;
            std::nth_element(log_samples, log_samples + log_idx_low, log_samples + sample_count);
            float log_min = log_samples[log_idx_low];
            std::nth_element(log_samples + log_idx_low + 1, log_samples + log_idx_high, log_samples + sample_count);
            float log_max = log_samples[log_idx_high];
            
            if (log_max <= log_min + 0.1f) {
                log_max = log_min + 1.0f;
            }
            
            // Normalize to 0-255 (TRUE METRIC: larger depth = brighter):
            // close objects (small metric depth -> small log) = DARK (0)
            // far objects (large metric depth -> large log) = BRIGHT (255)
            // Formula: output = 255 * (log_val - log_min) / (log_max - log_min)
            double log_range = log_max - log_min;
            double log_scale = 255.0 / log_range;
            double log_offset = -255.0 * log_min / log_range;
            log_depth.convertTo(depth_small_8u, CV_8U, log_scale, log_offset);
            
        } else {
            // --- INVERSE DEPTH VISUALIZATION (Original method) ---
            // Higher inverse depth = closer object = brighter
            double scale = 255.0 / (robustMax - robustMin);
            inv_depth.convertTo(depth_small_8u, CV_8U, scale, -robustMin * scale);
        }

        // [OPTIMIZATION D]: Resize the Byte Image (Fastest Resize)
        cv::resize(depth_small_8u, depth_8u, _img.size(), 0, 0, cv::INTER_LINEAR);
        
        pubDepthTrackImage(depth_8u, t);
        // [OPTIMIZATION E]: Single Channel Blending (No Merge/Split/CvtColor)
        // Weighted sum: 85% Original Gray + 15% Depth Map
        float coeff = nonlinear_input_cnt / 100.0f;
        cv::addWeighted(_img, 1 - coeff, depth_8u, coeff, 0, img_for_tracker);

        double time_proc = t_proc.toc();

        
        
        printf("[Depth] Infer: %.2f ms | Post-Proc: %.2f ms | Total Add: %.2f ms\n", 
                time_infer, time_proc, time_infer + time_proc);
        
    } 
  // Else: img_for_tracker remains the original _img (Mono8)

  // 4. Feature Tracking
  // Now strictly passing a Mono8 image every time
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
  
  if (params.multiple_thread) {
    mBuf.lock();
    while (!outlierBuf.empty()) {
      auto removeIndex = outlierBuf.front();
      outlierBuf.pop();
      featureTracker.removeOutliers(removeIndex);
    }
    while (!predictBuf.empty()) {
      auto preds = predictBuf.front();
      predictBuf.pop();
      featureTracker.setPrediction(preds);
    }
    mBuf.unlock();
  }

  if (_img1.empty())
    if (params.use_cuda_in_tracking)
      featureFrame = featureTracker.trackImageCUDA(t, img_for_tracker);
    else
      featureFrame = featureTracker.trackImage(t, img_for_tracker);
  else
    featureFrame = featureTracker.trackImage(t, img_for_tracker, _img1);

  if (params.show_track) {
    cv::Mat imgTrack = featureTracker.getTrackImage();
    pubTrackImage(imgTrack, t);
  }

  if (params.multiple_thread) {
    if (inputImageCnt % 2 == 0) {
      mBuf.lock();
      featureBuf.emplace(t, featureFrame);
      mBuf.unlock();
    }
  } else {
    mBuf.lock();
    featureBuf.emplace(t, featureFrame);
    mBuf.unlock();
    TicToc processTime;
    processMeasurements();
    printf("process time: %f\n", processTime.toc());
  }
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration,
                         const Vector3d &angularVelocity) {
  mBuf.lock();
  accBuf.emplace(t, linearAcceleration);
  gyrBuf.emplace(t, angularVelocity);
  // printf("input imu with time %f \n", t);
  mBuf.unlock();

  if (solver_flag == NON_LINEAR) {
    mPropagate.lock();
    fastPredictIMU(t, linearAcceleration, angularVelocity);
    pubLatestOdometry(latest_P, latest_Q, latest_V, t);
    mPropagate.unlock();
  }
}

void Estimator::inputFeature(
    double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
                  &featureFrame) {
  mBuf.lock();
  featureBuf.emplace(t, featureFrame);
  mBuf.unlock();

  if (!params.multiple_thread) processMeasurements();
}

bool Estimator::getIMUInterval(
    double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
    vector<pair<double, Eigen::Vector3d>> &gyrVector) {
  if (accBuf.empty()) {
    printf("not receive imu\n");
    return false;
  }
  // printf("get imu from %f %f\n", t0, t1);
  // printf("imu fornt time %f   imu end time %f\n", accBuf.front().first,
  // accBuf.back().first);
  if (t1 <= accBuf.back().first) {
    while (accBuf.front().first <= t0) {
      accBuf.pop();
      gyrBuf.pop();
    }
    while (accBuf.front().first < t1) {
      accVector.push_back(accBuf.front());
      accBuf.pop();
      gyrVector.push_back(gyrBuf.front());
      gyrBuf.pop();
    }
    accVector.push_back(accBuf.front());
    gyrVector.push_back(gyrBuf.front());
  } else {
    printf("wait for imu\n");
    return false;
  }
  return true;
}

bool Estimator::IMUAvailable(double t) {
  return !accBuf.empty() && t <= accBuf.back().first;
}

void Estimator::processMeasurements() {
  while (ros::ok()) {
    // printf("process measurments\n");
    pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>>
        feature;
    vector<pair<double, Eigen::Vector3d>> accVector;
    vector<pair<double, Eigen::Vector3d>> gyrVector;
    if (!featureBuf.empty()) {
      feature = featureBuf.front();
      curTime = feature.first + td;
      while (ros::ok()) {
        if ((!params.use_imu || IMUAvailable(feature.first + td))) break;
        printf("wait for imu ... \n");
        if (!params.multiple_thread) return;
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
      }
      mBuf.lock();
      if (params.use_imu)
        getIMUInterval(prevTime, curTime, accVector, gyrVector);

      featureBuf.pop();
      mBuf.unlock();

      if (params.use_imu) {
        if (!initFirstPoseFlag) initFirstIMUPose(accVector);
        for (size_t i = 0; i < accVector.size(); i++) {
          double dt;
          if (i == 0)
            dt = accVector[i].first - prevTime;
          else if (i == accVector.size() - 1)
            dt = curTime - accVector[i - 1].first;
          else
            dt = accVector[i].first - accVector[i - 1].first;
          processIMU(accVector[i].first, dt, accVector[i].second,
                     gyrVector[i].second);
        }
      }
      mProcess.lock();
      processImage(feature.second, feature.first);
      prevTime = curTime;

      printStatistics(*this, 0);

      std_msgs::Header header;
      header.frame_id = "world";
      header.stamp = ros::Time(feature.first);

      pubOdometry(*this, header);
      pubKeyPoses(*this, header);
      pubCameraPose(*this, header);
      pubPointCloud(*this, header);
      pubKeyframe(*this);
      pubTF(*this, header);
      mProcess.unlock();
    }

    if (!params.multiple_thread) break;

    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
}

void Estimator::initFirstIMUPose(
    vector<pair<double, Eigen::Vector3d>> &accVector) {
  printf("init first imu pose\n");
  initFirstPoseFlag = true;
  // return;
  Eigen::Vector3d averAcc(0, 0, 0);
  int n = static_cast<int>(accVector.size());
  for (auto &i : accVector) {
    averAcc = averAcc + i.second;
  }
  averAcc = averAcc / n;
  printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
  Matrix3d R0 = Utility::g2R(averAcc);
  double yaw = Utility::R2ypr(R0).x();
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  Rs[0] = R0;
  cout << "init R0 " << endl << Rs[0] << endl;
  // Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(const Eigen::Vector3d &p,
                              const Eigen::Matrix3d &r) {
  Ps[0] = p;
  Rs[0] = r;
  initP = p;
  initR = r;
}

void Estimator::processIMU(double /*t*/, double dt,
                           const Vector3d &linear_acceleration,
                           const Vector3d &angular_velocity) {
  if (!first_imu) {
    first_imu = true;
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
  }

  if (!pre_integrations[frame_count]) {
    pre_integrations[frame_count] = new IntegrationBase{
        acc_0,        gyr_0,        Bas[frame_count], Bgs[frame_count],
        params.acc_n, params.gyr_n, params.acc_w,     params.gyr_w,
        params.g};
  }
  if (frame_count != 0) {
    pre_integrations[frame_count]->push_back(dt, linear_acceleration,
                                             angular_velocity);
    // if(solver_flag != NON_LINEAR)
    tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

    dt_buf[frame_count].push_back(dt);
    linear_acceleration_buf[frame_count].push_back(linear_acceleration);
    angular_velocity_buf[frame_count].push_back(angular_velocity);

    int j = frame_count;
    Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
    Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
    Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
    Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
    Vs[j] += dt * un_acc;
  }
  acc_0 = linear_acceleration;
  gyr_0 = angular_velocity;
}

double Estimator::checkGeometricConsistency(const FeaturePerId &it_per_id, double proposed_metric_depth) {
    // 1. Get the latest observation (Current Frame)
    // We want to project FROM current TO start (or vice versa). 
    // VINS features are anchored in 'start_frame'.
    int start_frame_idx = it_per_id.start_frame;
    int current_frame_idx = it_per_id.endFrame();
    
    // If feature only exists in one frame, we can't verify geometry
    if (start_frame_idx == current_frame_idx) return 0.0;

    // 2. Get 2D observations
    Vector3d pt_start = it_per_id.feature_per_frame[0].point; // Unit sphere (x,y,z)
    Vector3d pt_curr  = it_per_id.feature_per_frame.back().point;

    // 3. Get Poses (World -> Body)
    // Ps[] and Rs[] are Body_T_World
    Vector3d P_start = Ps[start_frame_idx];
    Matrix3d R_start = Rs[start_frame_idx];
    Vector3d P_curr  = Ps[current_frame_idx];
    Matrix3d R_curr  = Rs[current_frame_idx];

    // 4. Extrinsics (Cam -> Body)
    Vector3d tic_0 = tic[0];
    Matrix3d ric_0 = ric[0];

    // 5. Back-project Start Point to World Frame using Proposed Depth
    // Point in Start Camera
    Vector3d pts_cam_start = pt_start * proposed_metric_depth;
    // Point in Body
    Vector3d pts_body_start = ric_0 * pts_cam_start + tic_0;
    // Point in World
    Vector3d pts_world = R_start * pts_body_start + P_start;

    // 6. Project World Point to Current Camera Frame
    // World -> Current Body
    Vector3d pts_body_curr = R_curr.transpose() * (pts_world - P_curr);
    // Current Body -> Current Camera
    Vector3d pts_cam_curr = ric_0.transpose() * (pts_body_curr - tic_0);

    // 7. Normalize (Project to Unit Plane)
    if (pts_cam_curr.z() <= 0.1) return 999.0; // Behind camera or too close
    Vector2d uv_projected = pts_cam_curr.head<2>() / pts_cam_curr.z();
    
    // 8. Compare with actual observation
    Vector2d uv_observed = pt_curr.head<2>() / pt_curr.z();
    
    double error = (uv_projected - uv_observed).norm();
    
    // Convert normalized coords to roughly pixels (assuming fx ~ 460)
    // This depends on your camera, but 460 is standard for RealSense/VINS
    return error * 1058.0; 
}

void Estimator::smartDepthInitialization() {
    // ------------------------------------------------------------------
    // STEP 0: PRE-CALCULATE THRESHOLDS (The Optimization)
    // ------------------------------------------------------------------
    // Map: Timestamp -> Sky Threshold
    TicToc t_smart;
    std::unordered_map<double, float> frame_sky_thresholds;

    // Iterate through all active frames in the window ONCE
    for (auto const& [timestamp, frame] : all_image_frame) {
        if (frame.depth_map.empty()) continue;

        // O(Pixels) operation happens only 10 times (Window Size) instead of 150+
        double min_val, max_val;
        cv::minMaxLoc(frame.depth_map, &min_val, &max_val);
        frame_sky_thresholds[timestamp] = (float)(min_val + 0.05 * (max_val - min_val));
    }

    // ------------------------------------------------------------------
    // STEP 1: LEARN (Use "The Teachers")
    // ------------------------------------------------------------------
    std::vector<double> v_vio_inv_depths;
    v_vio_inv_depths.reserve(100); // Reserve memory to avoid re-allocations
    std::vector<double> v_mono_inv_depths;
    v_mono_inv_depths.reserve(100);
    
    int valid_count = 0;
    double blind_guess_val = (params.init_depth > 0) ? params.init_depth : 5.0;

    for (auto &it_per_id : f_manager.feature) {
        if (it_per_id.feature_per_frame.size() < 4) continue;

        bool is_blind_guess = std::abs(it_per_id.estimated_depth - blind_guess_val) < 1e-4;
        
        if (it_per_id.estimated_depth > 0 && !is_blind_guess) {
            
            int first_frame_idx = it_per_id.start_frame;
            if (first_frame_idx >= WINDOW_SIZE + 1) continue; // Safety bounds
            
            double timestamp = Headers[first_frame_idx];

            // FAST LOOKUP: Check if we have a threshold for this frame
            auto thresh_it = frame_sky_thresholds.find(timestamp);
            if (thresh_it == frame_sky_thresholds.end()) continue;
            float sky_threshold = thresh_it->second;

            // Direct map access is risky if not found, use find() or ensure existence
            if (all_image_frame.find(timestamp) == all_image_frame.end()) continue;
            ImageFrame &frame = all_image_frame[timestamp];
            
            auto feature_data_it = frame.points.find(it_per_id.feature_id);
            if (feature_data_it == frame.points.end()) continue;

            const auto& measurement = feature_data_it->second[0].second;
            // Cast to int immediately
            int u = (int)measurement(3); 
            int v = (int)measurement(4); 

            // Fast boundary check
            if (u >= 0 && u < frame.depth_map.cols && v >= 0 && v < frame.depth_map.rows) {
                 float d_mono_inv = frame.depth_map.at<float>(v, u);

                 // Use pre-calculated threshold
                 if (d_mono_inv > sky_threshold) {
                     v_vio_inv_depths.push_back(1.0 / it_per_id.estimated_depth);
                     v_mono_inv_depths.push_back(d_mono_inv);
                     valid_count++;
                 }
            }
        }
    }

    // ------------------------------------------------------------------
    // STEP 2: CALCULATE (RANSAC Robust Fitting)
    // ------------------------------------------------------------------
    if (valid_count > 20) { 
        
        int best_inlier_count = -1;
        double best_s = cached_scale; // Default to previous knowledge
        double best_t = cached_shift;

        // RANSAC Parameters
        const int iterations = 50; 
        const double threshold = 0.05; // Tolerance in Inverse Depth units
        const int num_points = v_mono_inv_depths.size();

        // Random generator setup
        std::srand(std::time(nullptr)); // Seed (or use standard C++ random engine)

        for (int k = 0; k < iterations; k++) {
            // 1. Pick 2 random unique points
            int idx1 = std::rand() % num_points;
            int idx2 = std::rand() % num_points;
            if (idx1 == idx2) continue; // Skip if same point picked twice

            double x1 = v_mono_inv_depths[idx1];
            double y1 = v_vio_inv_depths[idx1];
            double x2 = v_mono_inv_depths[idx2];
            double y2 = v_vio_inv_depths[idx2];

            // 2. Compute Model (y = s*x + t) from these 2 points
            double denom = x1 - x2;
            if (std::abs(denom) < 1e-6) continue; // Avoid vertical lines

            double s_candidate = (y1 - y2) / denom;
            double t_candidate = y1 - s_candidate * x1;

            // Optional: Constraint Check
            // We expect scale to be positive (depth correlates with depth)
            if (s_candidate <= 0) continue; 

            // 3. Count Inliers
            int current_inliers = 0;
            for (int i = 0; i < num_points; i++) {
                double error = std::abs(v_vio_inv_depths[i] - (s_candidate * v_mono_inv_depths[i] + t_candidate));
                if (error < threshold) {
                    current_inliers++;
                }
            }

            // 4. Update Best Model
            if (current_inliers > best_inlier_count) {
                best_inlier_count = current_inliers;
                best_s = s_candidate;
                best_t = t_candidate;
            }
        }

        // [OPTIONAL BUT RECOMMENDED] Refinement:
        // Re-run Least Squares ONLY on the inliers of the best model 
        // to get the most precise fit.
        if (best_inlier_count > 12) {
            double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
            int refined_n = 0;
            
            for (int i = 0; i < num_points; i++) {
                double error = std::abs(v_vio_inv_depths[i] - (best_s * v_mono_inv_depths[i] + best_t));
                if (error < threshold) {
                    sum_x += v_mono_inv_depths[i];
                    sum_y += v_vio_inv_depths[i];
                    sum_xy += v_mono_inv_depths[i] * v_vio_inv_depths[i];
                    sum_xx += v_mono_inv_depths[i] * v_mono_inv_depths[i];
                    refined_n++;
                }
            }
            
            double denom = (refined_n * sum_xx - sum_x * sum_x);
            if (std::abs(denom) > 1e-6) {
                best_s = (refined_n * sum_xy - sum_x * sum_y) / denom;
                best_t = (sum_y - best_s * sum_x) / refined_n;
            }
        }

        // 5. Update Global State with Smoothing
        if (!scale_is_initialized) {
            cached_scale = best_s;
            cached_shift = best_t;
            scale_is_initialized = true;
        } else {
            // Lower alpha (0.1) because RANSAC can jump a bit more than LS
            double alpha = 0.1; 
            cached_scale = (1.0 - alpha) * cached_scale + alpha * best_s;
            cached_shift = (1.0 - alpha) * cached_shift + alpha * best_t;
        }
        
        // ------------------------------------------------------------------
        // STEP 2.5: COMPUTE VARIANCE for Mahalanobis weighting (in inverse domain)
        // ------------------------------------------------------------------
        // Using INLIERS only to compute robust variance estimate
        double sum_errors = 0.0;
        double sum_errors_sq = 0.0;
        int variance_n = 0;
        
        for (int i = 0; i < num_points; i++) {
            double aligned_inv = best_s * v_mono_inv_depths[i] + best_t;
            double error = v_vio_inv_depths[i] - aligned_inv;
            double abs_error = std::abs(error);
            
            // Only use inliers for variance computation (same threshold as RANSAC)
            if (abs_error < threshold) {
                sum_errors += error;
                sum_errors_sq += error * error;
                variance_n++;
            }
        }
        
        if (variance_n > 5) {
            double mean_error = sum_errors / variance_n;
            double variance = (sum_errors_sq / variance_n) - (mean_error * mean_error);
            
            // Ensure minimum variance to avoid numerical issues
            variance = std::max(variance, 1e-6);
            
            // Exponential smoothing for variance (slower update to be stable)
            double var_alpha = 0.05;
            if (!scale_is_initialized || cached_inv_depth_variance < 1e-8) {
                cached_inv_depth_variance = variance;
                cached_inv_depth_mean_error = mean_error;
            } else {
                cached_inv_depth_variance = (1.0 - var_alpha) * cached_inv_depth_variance + var_alpha * variance;
                cached_inv_depth_mean_error = (1.0 - var_alpha) * cached_inv_depth_mean_error + var_alpha * mean_error;
            }
            
            printf("\033[1;33m[Depth Mahalanobis] Inv-depth variance=%.6f, mean_err=%.6f, n=%d\033[0m\n",
                   cached_inv_depth_variance, cached_inv_depth_mean_error, variance_n);
        }
    }

    // ------------------------------------------------------------------
    // STEP 3: RESCUE (Fix "The Students")
    // ------------------------------------------------------------------
    int rescued_count = 0;

    TicToc t_rescue;
    if (scale_is_initialized) {
        for (auto &it_per_id : f_manager.feature) {
            if (it_per_id.feature_per_frame.size() < 2) continue;

            bool is_blind_guess = std::abs(it_per_id.estimated_depth - blind_guess_val) < 1e-4;

            if (is_blind_guess) {
                int first_frame_idx = it_per_id.start_frame;
                double timestamp = Headers[first_frame_idx];
                
                // FAST LOOKUP
                auto thresh_it = frame_sky_thresholds.find(timestamp);
                if (thresh_it == frame_sky_thresholds.end()) continue;
                float sky_threshold = thresh_it->second;
                
                if (all_image_frame.find(timestamp) == all_image_frame.end()) continue;
                ImageFrame &frame = all_image_frame[timestamp];

                auto feature_data_it = frame.points.find(it_per_id.feature_id);
                if (feature_data_it == frame.points.end()) continue;

                const auto& measurement = feature_data_it->second[0].second;
                int x_raw = (int)measurement(3);
                int y_raw = (int)measurement(4);

                if (x_raw >= 0 && x_raw < frame.depth_map.cols && y_raw >= 0 && y_raw < frame.depth_map.rows) {
                    
                    int x0 = static_cast<int>(std::floor(x_raw));
              int y0 = static_cast<int>(std::floor(y_raw));
              cv::Mat& depth_map = frame.depth_map;
              //select a 6x6 neighborhood's max value
              float depth_net_val = -1.0f;
              float depth_net_initial = depth_map.at<float>(y0, x0);
              for (int dx = -2; dx <= 2; dx++){
                for (int dy = -2; dy <= 2; dy++){
                  int nx = x0 + dx;
                  int ny = y0 + dy;
                  if (nx >= 0 && nx < depth_map.cols && ny >=0 && ny < depth_map.rows){
                    float val = depth_map.at<float>(ny, nx);
                    if (val > depth_net_val) {
                      depth_net_val = val; 
                      
                    }
            }
          }
        }
          if (std::abs(1 - (depth_net_initial / depth_net_val)) < 0.2) {
            depth_net_val = depth_net_initial; 
            
          } 
          else{
            std::cout << "Using highest inverse depth in neighborhood: " << depth_net_val << " instead of " << depth_net_initial << std::endl;
          }
                    float d_mono_inv = depth_net_val;
                    if (d_mono_inv > sky_threshold) {
                         double pred_inv_depth = cached_scale * d_mono_inv + cached_shift;
                         
                         if (pred_inv_depth > 0.01) { 
                             double new_depth = 1.0 / pred_inv_depth;
                             
                             if(0.01 < new_depth < 40.0) {
                              double reproj_err = checkGeometricConsistency(it_per_id, new_depth);
                              if (reproj_err > 20.0) {
                                 printf("[Zombie Killed] ID %d Rejected. Depth %.2fm caused %.2f px error.\n", 
                        it_per_id.feature_id, new_depth, reproj_err);
                                continue; // Reject if reprojection error too high
                              }
                                 it_per_id.estimated_depth = new_depth;
                                 it_per_id.solve_flag = 1; 
                                 rescued_count++;
                             }
                         }
                    }
                }
            }
        }
    }
    if (scale_is_initialized) ROS_INFO("[Rescue] Time cost: %f ms", t_rescue.toc());
    
    if (rescued_count > 0 || (valid_count > 0 && frame_count % 15 == 0)) {
        
        // Green text for visibility
        printf("\033[1;32m[Smart Init] Learned s=%.4f, t=%.4f (based on %d features) | RESCUED %d features!\033[0m\n", 
               cached_scale, cached_shift, valid_count, rescued_count);
    }
}

void Estimator::processImage(
    const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
    const double header) {
  ROS_DEBUG("new image coming ------------------------------------------");
  ROS_DEBUG("Adding feature points %lu", image.size());
  if (f_manager.addFeatureCheckParallax(frame_count, image, td)) {
    marginalization_flag = MARGIN_OLD;
    //in parallel
    // MatchResult matches = sp_lg_inferer->run(prev_img, curr_img);
    // printf("keyframe\n");
  } else {
    marginalization_flag = MARGIN_SECOND_NEW;
    // printf("non-keyframe\n");
  }
  // save the feature points to feature_debug_path
  if (params.feature_debug) {
    vins::estimator::FeatureManager::logFeature(image,
                                                params.feature_debug_path);
  }
  cv::Mat current_depth;
  ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
  ROS_DEBUG("Solving %d", frame_count);
  ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
  std::cout << "Use depth: " << params.use_depth << ", use gt depth: " << params.use_gt << "weight: " << WEIGHT << std::endl;
  if (marginalization_flag == MARGIN_OLD && params.use_depth && WEIGHT > 0.0) {
        std::cout << header << " is a keyframe, lets get the depth!" << std::endl;
        // 1. Retrieve the image from our cache
        cv::Mat raw_img;
        bool found = false;
        
        //mCache.lock();
        //auto it = image_cache.lower_bound(header - 0.001); 
        //
        //if (it != image_cache.end() && std::abs(it->first - header) < 0.001) {
        //    // Found a match within 1ms tolerance
        //    raw_img = it->second;
        //    found = true;
        //    // But if you do erase, make sure you erase the iterator, not the double key
        //    image_cache.erase(it); 
        //}
        //mCache.unlock();

        mCache.lock(); // Lock once for the batch operation

        // Iterate through the active VINS window (Indices 0 to WINDOW_SIZE-1)
        // We skip WINDOW_SIZE (Index 10) because it's volatile.
        for (int i = 0; i < WINDOW_SIZE-1; i++) {

            double ts = Headers[i];

            // 1. Check if this frame exists in our map
            if (all_image_frame.find(ts) == all_image_frame.end()) continue;

            ImageFrame &frame = all_image_frame.at(ts);
            
            // 2. If it has no depth, it needs it NOW (it has survived long enough)
            if (frame.depth_map.empty() && params.use_depth) {

                // 3. Check if we still have the raw image in cache
                if (image_cache.count(ts)) {
                    ROS_INFO("Lazy Inference: Backfilling depth for Frame %.6f (Index %d)", ts, i);

                    cv::Mat raw_depth_518;
                    
                    if (!params.use_gt) {
                        // Run neural network inference
                        cv::Mat raw_img = image_cache[ts];
                        mCache.unlock(); 
                        raw_depth_518 = depthInferer->infer(raw_img);
                        mCache.lock();
                    } else {
                        // Load ground truth depth from file
                        int frame_idx = frame_index_cache.count(ts) ? frame_index_cache[ts] : -1;
                        if (frame_idx >= 0) {
                            char frame_str[16];
                            snprintf(frame_str, sizeof(frame_str), "%06d", frame_idx);
                            std::string gt_depth_path = params.depth_folder + "/" + std::string(frame_str) + "_lcam_front_depth.tiff";
                            
                            cv::Mat gt_depth = cv::imread(gt_depth_path, cv::IMREAD_UNCHANGED);
                            if (!gt_depth.empty()) {
                                // Convert metric depth to inverse depth
                                cv::divide(1.0, gt_depth, raw_depth_518, 1.0, CV_32F);
                                
                                std::cout << "Loaded ground truth depth image from " << gt_depth_path << std::endl;
                              
                            } else {
                                ROS_WARN("Could not load GT depth from %s", gt_depth_path.c_str());
                                continue;
                            }
                        } else {
                            ROS_WARN("No frame index found for timestamp %.6f", ts);
                            continue;
                        }
                    }

                    cv::Mat resized_depth;
                    cv::resize(raw_depth_518, resized_depth, cv::Size(params.col, params.row));

                    frame.depth_map = resized_depth.clone();
                      if (!frame.depth_map.empty()) {
                        // 1. Update the tracker with the REAL metric depth (don't normalize this!)
                        

                        // 2. Create a separate image just for visualization
                        cv::Mat depth_vis;
                        
                        // Normalize: Map min_depth -> 0 and max_depth -> 255
                        cv::normalize(resized_depth, depth_vis, 0, 255, cv::NORM_MINMAX);
                        
                        // Convert to 8-bit (standard image format)
                        depth_vis.convertTo(depth_vis, CV_8UC1);

                        // Optional: Apply a colormap (makes it easier to see relative depth)
                        // cv::Mat depth_color;
                        // cv::applyColorMap(depth_vis, depth_color, cv::COLORMAP_JET);
                        //featureTracker.updateDepth(depth_vis);
                        // 3. Save the visualization
                        // Make sure you created the folder: mkdir -p /root/catkin_ws/debug_images
                        
                        //cv::Mat img = featureTracker.getTrackImage();
                        //if (!img.empty() && !depth_vis.empty()){
                        //    cv::imshow("RGB Track", img);
                        //    cv::imshow("Depth Track", depth_vis);
                        //    cv::waitKey(1);
                        //}
                        
                        //if (inputImageCnt % 20 == 0) saveImageToFolder(depthTrack, "/datasets/vins_debug/", "example_rgb.png");

                        pubDepthTrackImage(depth_vis, ts);
                      }
                    // NOW we can delete it from cache, we're done with it
                    image_cache.erase(ts); 
                } else {
                     ROS_WARN("Lazy Inference Failed: RGB image for Frame %.6f missing from cache!", ts);
                }
            } else {
                // If it already has depth, we can ensure the raw image is cleared to save RAM
                if (image_cache.count(ts)) image_cache.erase(ts);
            }
        }

        // Cleanup: Ensure cache doesn't hold images older than the oldest window frame
        double oldest_time = Headers[0];
        for(auto it = image_cache.begin(); it != image_cache.end(); ) {
            if(it->first < oldest_time - 0.5) { // 0.5s buffer
                it = image_cache.erase(it);
            } else {
                ++it;
            }
        }

        mCache.unlock();
        // 2. Run Inference
        if (found && params.use_depth && WEIGHT>0.0) {
        // 1. Run Inference (Returns 518x518)
        cv::Mat raw_depth_518 = depthInferer->infer(raw_img);
        // 2. Resize to match VINS frame (1280x720)
        // VINS expects features coordinates in the original resolution
        cv::resize(raw_depth_518, current_depth, cv::Size(params.col, params.row));
        
        // --- DEBUG BLOCK START ---
        //double minVal, maxVal;
        //cv::minMaxLoc(current_depth, &minVal, &maxVal);
        //cv::Scalar avgVal = cv::mean(current_depth);
        //ROS_INFO_STREAM("Depth Debug:"
        //    << " Size=" << current_depth.cols << "x" << current_depth.rows
        //    << " | Type=" << current_depth.type()  // Should be 5 (CV_32FC1)
        //    << " | Min=" << minVal
        //    << " | Max=" << maxVal
        //    << " | Avg=" << avgVal[0]);
        // --- DEBUG BLOCK END ---
        } else {
            if (params.use_depth) ROS_WARN("Keyframe image not found in cache! Timestamp mismatch?");
        }
    }
  Headers[frame_count] = header;
  ImageFrame imageframe(image, header);
  imageframe.pre_integration = tmp_pre_integration;
  if (marginalization_flag == MARGIN_OLD) {
      imageframe.is_optimization_keyframe = true;
  } else {
      imageframe.is_optimization_keyframe = false;
  }
  // Set absolute frame index from cache for GT depth loading
  if (frame_index_cache.count(header)) {
      imageframe.frame_index = frame_index_cache[header];
  }
  auto insertion_result = all_image_frame.insert(make_pair(header, imageframe));
  auto& map_frame_ref = insertion_result.first->second;
  //all_image_frame.insert(make_pair(header, imageframe));
  tmp_pre_integration = new IntegrationBase{
      acc_0,        gyr_0,        Bas[frame_count], Bgs[frame_count],
      params.acc_n, params.gyr_n, params.acc_w,     params.gyr_w,
      params.g};
  
  if (!current_depth.empty()) {
        map_frame_ref.depth_map = current_depth.clone();
        cv::Mat grid_view;
    // Resize 1280x800 -> 32x20
    // INTER_AREA is best for decimation (downsampling) as it respects pixel area relations
    cv::resize(current_depth, grid_view, cv::Size(32, 20), 0, 0, cv::INTER_AREA);

    std::cout << "\n========== 32x20 DEPTH GRID ==========\n";
    std::cout << std::fixed << std::setprecision(2); // Fix float formatting to 2 decimals

    for (int r = 0; r < grid_view.rows; ++r) {
        for (int c = 0; c < grid_view.cols; ++c) {
            float val = grid_view.at<float>(r, c);
            // setw(5) ensures columns align nicely (e.g. " 1.25")
            std::cout << std::setw(5) << val << " ";
        }
        std::cout << "\n"; // Newline at the end of every row
    }
    std::cout << "======================================\n" << std::endl;
        ROS_INFO("Confirmed: Depth map attached to frame %.6f", header);
    }
  if (params.estimate_extrinsic == 2) {
    ROS_INFO("calibrating extrinsic param, rotation movement is needed");
    if (frame_count != 0) {
      vector<pair<Vector3d, Vector3d>> corres =
          f_manager.getCorresponding(frame_count - 1, frame_count);
      Matrix3d calib_ric;
      if (initial_ex_rotation.CalibrationExRotation(
              corres, pre_integrations[frame_count]->delta_q, calib_ric)) {
        ROS_WARN("initial extrinsic rotation calib success");
        ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
        ric[0] = calib_ric;
        params.ric[0] = calib_ric;
        params.estimate_extrinsic = 1;
      }
    }
  }

  if (solver_flag == INITIAL) {
    // monocular + IMU initilization
    if (!params.stereo && params.use_imu) {
      if (frame_count == WINDOW_SIZE) {
        bool result = false;
        if (params.estimate_extrinsic != 2 &&
            (header - initial_timestamp) > 0.1) {
          result = initialStructure();
          initial_timestamp = header;
        }
        if (result) {
          optimization();
          updateLatestStates();
          solver_flag = NON_LINEAR;
          slideWindow();
          ROS_INFO("Initialization finish!");
        } else
          slideWindow();
      }
    }

    // stereo + IMU initilization
    if (params.stereo && params.use_imu) {
      f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
      f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
      if (frame_count == WINDOW_SIZE) {
        map<double, ImageFrame>::iterator frame_it;
        int i = 0;
        for (frame_it = all_image_frame.begin();
             frame_it != all_image_frame.end(); frame_it++) {
          frame_it->second.R = Rs[i];
          frame_it->second.T = Ps[i];
          i++;
        }
        solveGyroscopeBias(all_image_frame, Bgs);
        for (int i = 0; i <= WINDOW_SIZE; i++) {
          pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
        }
        optimization();
        updateLatestStates();
        solver_flag = NON_LINEAR;
        slideWindow();
        ROS_INFO("Initialization finish!");
      }
    }

    // stereo only initilization
    if (params.stereo && !params.use_imu) {
      f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
      f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
      optimization();

      if (frame_count == WINDOW_SIZE) {
        optimization();
        updateLatestStates();
        solver_flag = NON_LINEAR;
        slideWindow();
        ROS_INFO("Initialization finish!");
      }
    }

    if (frame_count < WINDOW_SIZE) {
      frame_count++;
      int prev_frame = frame_count - 1;
      Ps[frame_count] = Ps[prev_frame];
      Vs[frame_count] = Vs[prev_frame];
      Rs[frame_count] = Rs[prev_frame];
      Bas[frame_count] = Bas[prev_frame];
      Bgs[frame_count] = Bgs[prev_frame];
    }

  } else {
    TicToc t_solve;
    if (params.stereo_init && params.stereo && stereo_init_counter > params.stereo_init_lag) {
      params.stereo = false;
    } else {
      stereo_init_counter++;
    }

    if (!params.use_imu)
      f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
    if (solver_flag == NON_LINEAR && params.use_depth) {
      TicToc t_depth_init;
      std::cout << "Before smart depth initialization: scale=" << cached_scale << ", shift=" << cached_shift << std::endl;
      smartDepthInitialization();
      ROS_INFO("smart depth initialization costs: %f ms", t_depth_init.toc());
    }
    optimization();
    set<int> removeIndex;
    outliersRejection(removeIndex);

    if (params.feature_debug) {
      vins::estimator::FeatureManager::logOutlier(removeIndex,
                                                  params.feature_debug_path);
    }

    f_manager.removeOutlier(removeIndex);
    if (params.tracking_outlier_rejection) {
      if (!params.multiple_thread) {
        featureTracker.removeOutliers(removeIndex);
      } else {
        mBuf.lock();
        outlierBuf.push(removeIndex);
        mBuf.unlock();
      }
    }

    if (params.tracking_prediction) {
      auto preds = predictPtsInNextFrame();
      if (!params.multiple_thread) {
        featureTracker.setPrediction(preds);
      } else {
        mBuf.lock();
        predictBuf.push(preds);
        mBuf.unlock();
      }
    }

    ROS_DEBUG("solver costs: %fms", t_solve.toc());

    if (failureDetection()) {
      ROS_WARN("failure detection!");
      failure_occur = true;
      clearState();
      setParameter();
      ROS_WARN("system reboot!");
      return;
    }

    slideWindow();
    f_manager.removeFailures();
    // prepare output of VINS
    key_poses.clear();
    for (int i = 0; i <= WINDOW_SIZE; i++) key_poses.push_back(Ps[i]);

    last_R = Rs[WINDOW_SIZE];
    last_P = Ps[WINDOW_SIZE];
    last_R0 = Rs[0];
    last_P0 = Ps[0];
    updateLatestStates();
  }
}

bool Estimator::initialStructure() {
  TicToc t_sfm;
  // check imu observibility
  {
    map<double, ImageFrame>::iterator frame_it;
    Vector3d sum_g;
    for (frame_it = all_image_frame.begin(), frame_it++;
         frame_it != all_image_frame.end(); frame_it++) {
      double dt = frame_it->second.pre_integration->sum_dt;
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      sum_g += tmp_g;
    }
    Vector3d aver_g;
    aver_g = sum_g * 1.0 / (static_cast<int>(all_image_frame.size()) - 1);
    double var = 0;
    for (frame_it = all_image_frame.begin(), frame_it++;
         frame_it != all_image_frame.end(); frame_it++) {
      double dt = frame_it->second.pre_integration->sum_dt;
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
      // cout << "frame g " << tmp_g.transpose() << endl;
    }
    var = sqrt(var / (static_cast<int>(all_image_frame.size()) - 1));
    // ROS_WARN("IMU variation %f!", var);
    if (var < 0.25) {
      ROS_INFO("IMU excitation not enouth!");
      // return false;
    }
  }
  // global sfm
  Quaterniond Q[frame_count + 1];
  Vector3d T[frame_count + 1];
  map<int, Vector3d> sfm_tracked_points;
  vector<SFMFeature> sfm_f;
  for (auto &it_per_id : f_manager.feature) {
    int imu_j = it_per_id.start_frame - 1;
    SFMFeature tmp_feature;
    tmp_feature.state = false;
    tmp_feature.id = it_per_id.feature_id;
    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      Vector3d pts_j = it_per_frame.point;
      tmp_feature.observation.push_back(
          make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
    }
    sfm_f.push_back(tmp_feature);
  }
  Matrix3d relative_R;
  Vector3d relative_T;
  int l;
  if (!relativePose(relative_R, relative_T, l)) {
    ROS_INFO("Not enough features or parallax; Move device around");
    return false;
  }
  GlobalSFM sfm{params};
  if (!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f,
                     sfm_tracked_points)) {
    ROS_DEBUG("global SFM failed!");
    marginalization_flag = MARGIN_OLD;
    return false;
  }

  // solve pnp for all frame
  map<double, ImageFrame>::iterator frame_it;
  map<int, Vector3d>::iterator it;
  frame_it = all_image_frame.begin();
  for (int i = 0; frame_it != all_image_frame.end(); frame_it++) {
    // provide initial guess
    cv::Mat r;
    cv::Mat rvec;
    cv::Mat t;
    cv::Mat D;
    cv::Mat tmp_r;
    if ((frame_it->first) == Headers[i]) {
      frame_it->second.is_key_frame = true;
      frame_it->second.R = Q[i].toRotationMatrix() * params.ric[0].transpose();
      frame_it->second.T = T[i];
      i++;
      continue;
    }
    if ((frame_it->first) > Headers[i]) {
      i++;
    }
    Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
    Vector3d P_inital = -R_inital * T[i];
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    frame_it->second.is_key_frame = false;
    vector<cv::Point3f> pts_3_vector;
    vector<cv::Point2f> pts_2_vector;
    for (auto &id_pts : frame_it->second.points) {
      int feature_id = id_pts.first;
      for (auto &i_p : id_pts.second) {
        it = sfm_tracked_points.find(feature_id);
        if (it != sfm_tracked_points.end()) {
          Vector3d world_pts = it->second;
          cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
          pts_3_vector.push_back(pts_3);
          Vector2d img_pts = i_p.second.head<2>();
          cv::Point2f pts_2(img_pts(0), img_pts(1));
          pts_2_vector.push_back(pts_2);
        }
      }
    }
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    if (pts_3_vector.size() < 6) {
      cout << "pts_3_vector size " << pts_3_vector.size() << endl;
      ROS_DEBUG("Not enough points for solve pnp !");
      return false;
    }
    if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true)) {
      ROS_DEBUG("solve pnp fail!");
      return false;
    }
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp;
    MatrixXd tmp_R_pnp;
    cv::cv2eigen(r, tmp_R_pnp);
    R_pnp = tmp_R_pnp.transpose();
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    T_pnp = R_pnp * (-T_pnp);
    frame_it->second.R = R_pnp * params.ric[0].transpose();
    frame_it->second.T = T_pnp;
  }
  if (visualInitialAlign()) return true;
  ROS_INFO("misalign visual structure with IMU");
  return false;
}

bool Estimator::visualInitialAlign() {
  TicToc t_g;
  VectorXd x;
  // solve scale
  bool result =
      VisualIMUAlignment(all_image_frame, Bgs, g, x, params.g, params.tic[0]);
  if (!result) {
    ROS_DEBUG("solve g failed!");
    return false;
  }

  // change state
  for (int i = 0; i <= frame_count; i++) {
    Matrix3d Ri = all_image_frame[Headers[i]].R;
    Vector3d Pi = all_image_frame[Headers[i]].T;
    Ps[i] = Pi;
    Rs[i] = Ri;
    all_image_frame[Headers[i]].is_key_frame = true;
  }

  double s = (x.tail<1>())(0);
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
  }
  for (int i = frame_count; i >= 0; i--)
    Ps[i] =
        s * Ps[i] - Rs[i] * params.tic[0] - (s * Ps[0] - Rs[0] * params.tic[0]);
  int kv = -1;
  map<double, ImageFrame>::iterator frame_i;
  for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end();
       frame_i++) {
    if (frame_i->second.is_key_frame) {
      kv++;
      Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
    }
  }

  Matrix3d R0 = Utility::g2R(g);
  double yaw = Utility::R2ypr(R0 * Rs[0]).x();
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  g = R0 * g;
  // Matrix3d rot_diff = R0 * Rs[0].transpose();
  Matrix3d rot_diff = R0;
  for (int i = 0; i <= frame_count; i++) {
    Ps[i] = rot_diff * Ps[i];
    Rs[i] = rot_diff * Rs[i];
    Vs[i] = rot_diff * Vs[i];
  }
  ROS_DEBUG_STREAM("g0     " << g.transpose());
  ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

  f_manager.clearDepth();
  f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

  return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T,
                             int &l) {
  // find previous frame which contians enough correspondance and parallex with
  // newest frame
  for (int i = 0; i < WINDOW_SIZE; i++) {
    vector<pair<Vector3d, Vector3d>> corres;
    corres = f_manager.getCorresponding(i, WINDOW_SIZE);
    if (corres.size() > 20) {
      double sum_parallax = 0;
      double average_parallax;
      for (auto &corre : corres) {
        Vector2d pts_0(corre.first(0), corre.first(1));
        Vector2d pts_1(corre.second(0), corre.second(1));
        double parallax = (pts_0 - pts_1).norm();
        sum_parallax = sum_parallax + parallax;
      }
      average_parallax = 1.0 * sum_parallax / static_cast<int>(corres.size());
      if (average_parallax * params.focal_length > 30 &&
          m_estimator.solveRelativeRT(corres, relative_R, relative_T,
                                      params.focal_length)) {
        l = i;
        ROS_DEBUG(
            "average_parallax %f choose l %d and newest frame to triangulate "
            "the whole structure",
            average_parallax * params.focal_length, l);
        return true;
      }
    }
  }
  return false;
}

void Estimator::vector2double() {
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    para_Pose[i][0] = Ps[i].x();
    para_Pose[i][1] = Ps[i].y();
    para_Pose[i][2] = Ps[i].z();
    Quaterniond q{Rs[i]};
    para_Pose[i][3] = q.x();
    para_Pose[i][4] = q.y();
    para_Pose[i][5] = q.z();
    para_Pose[i][6] = q.w();

    if (params.use_imu) {
      para_SpeedBias[i][0] = Vs[i].x();
      para_SpeedBias[i][1] = Vs[i].y();
      para_SpeedBias[i][2] = Vs[i].z();

      para_SpeedBias[i][3] = Bas[i].x();
      para_SpeedBias[i][4] = Bas[i].y();
      para_SpeedBias[i][5] = Bas[i].z();

      para_SpeedBias[i][6] = Bgs[i].x();
      para_SpeedBias[i][7] = Bgs[i].y();
      para_SpeedBias[i][8] = Bgs[i].z();
    }
  }

  for (int i = 0; i < params.num_of_cam; i++) {
    para_Ex_Pose[i][0] = tic[i].x();
    para_Ex_Pose[i][1] = tic[i].y();
    para_Ex_Pose[i][2] = tic[i].z();
    Quaterniond q{ric[i]};
    para_Ex_Pose[i][3] = q.x();
    para_Ex_Pose[i][4] = q.y();
    para_Ex_Pose[i][5] = q.z();
    para_Ex_Pose[i][6] = q.w();
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < f_manager.getFeatureCount(); i++) {
    para_Feature[i][0] = dep(i);
    assert(i < NUM_OF_F);
  }
  for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_ScaleShift[i][0] = depth_net_scales[i];
        para_ScaleShift[i][1] = depth_net_shifts[i];
    }
  para_Td[0][0] = td;
}

void Estimator::double2vector() {
  Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
  Vector3d origin_P0 = Ps[0];

  if (failure_occur) {
    origin_R0 = Utility::R2ypr(last_R0);
    origin_P0 = last_P0;
    failure_occur = false;
  }

  if (params.use_imu) {
    Vector3d origin_R00 =
        Utility::R2ypr(Quaterniond(para_Pose[0][6], para_Pose[0][3],
                                   para_Pose[0][4], para_Pose[0][5])
                           .toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    // TODO(unknown):
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 ||
        abs(abs(origin_R00.y()) - 90) < 1.0) {
      ROS_DEBUG("euler singular point!");
      rot_diff = Rs[0] * Quaterniond(para_Pose[0][6], para_Pose[0][3],
                                     para_Pose[0][4], para_Pose[0][5])
                             .toRotationMatrix()
                             .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++) {
      Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3],
                                     para_Pose[i][4], para_Pose[i][5])
                             .normalized()
                             .toRotationMatrix();

      Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                  para_Pose[i][1] - para_Pose[0][1],
                                  para_Pose[i][2] - para_Pose[0][2]) +
              origin_P0;

      Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0], para_SpeedBias[i][1],
                                  para_SpeedBias[i][2]);

      Bas[i] = Vector3d(para_SpeedBias[i][3], para_SpeedBias[i][4],
                        para_SpeedBias[i][5]);

      Bgs[i] = Vector3d(para_SpeedBias[i][6], para_SpeedBias[i][7],
                        para_SpeedBias[i][8]);
    }
  } else {
    for (int i = 0; i <= WINDOW_SIZE; i++) {
      Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4],
                          para_Pose[i][5])
                  .normalized()
                  .toRotationMatrix();

      Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
    }
  }

  if (params.use_imu) {
    for (int i = 0; i < params.num_of_cam; i++) {
      tic[i] =
          Vector3d(para_Ex_Pose[i][0], para_Ex_Pose[i][1], para_Ex_Pose[i][2]);
      ric[i] = Quaterniond(para_Ex_Pose[i][6], para_Ex_Pose[i][3],
                           para_Ex_Pose[i][4], para_Ex_Pose[i][5])
                   .normalized()
                   .toRotationMatrix();
    }
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < f_manager.getFeatureCount(); i++) {
    dep(i) = para_Feature[i][0];
    assert(i < NUM_OF_F);
  }
  f_manager.setDepth(dep);
  for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        depth_net_scales[i] = para_ScaleShift[i][0];
        depth_net_shifts[i] = para_ScaleShift[i][1];
    }
  if (params.use_imu) td = para_Td[0][0];
}

bool Estimator::failureDetection() {
  return false;
  if (f_manager.last_track_num < 2) {
    ROS_INFO(" little feature %d", f_manager.last_track_num);
    // return true;
  }
  if (Bas[WINDOW_SIZE].norm() > 2.5) {
    ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
    return true;
  }
  if (Bgs[WINDOW_SIZE].norm() > 1.0) {
    ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
    return true;
  }
  /*
  if (tic(0) > 1)
  {
      ROS_INFO(" big extri param estimation %d", tic(0) > 1);
      return true;
  }
  */
  Vector3d tmp_P = Ps[WINDOW_SIZE];
  if ((tmp_P - last_P).norm() > 5) {
    // ROS_INFO(" big translation");
    // return true;
  }
  if (abs(tmp_P.z() - last_P.z()) > 1) {
    // ROS_INFO(" big z translation");
    // return true;
  }
  Matrix3d tmp_R = Rs[WINDOW_SIZE];
  Matrix3d delta_R = tmp_R.transpose() * last_R;
  Quaterniond delta_Q(delta_R);
  double delta_angle;
  delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
  if (delta_angle > 50) {
    ROS_INFO(" big delta_angle ");
    // return true;
  }
  return false;
}

// ============================================================================
// OPTIMIZATION DIAGNOSTICS STRUCT - Tracks factor counts, residuals, and costs
// ============================================================================
struct FeatureCostInfo {
  int feature_id;
  int feature_index;
  int start_frame;
  int num_observations;
  double inv_depth_before;
  double inv_depth_after;
  double cost_before;
  double cost_after;
  std::vector<ceres::ResidualBlockId> block_ids;
};

struct OptimizationDiagnostics {
  // Factor counts
  int num_marginalization_factors = 0;
  int num_imu_factors = 0;
  int num_reprojection_mono_factors = 0;      // ProjectionTwoFrameOneCamFactor
  int num_reprojection_stereo_factors = 0;    // ProjectionTwoFrameTwoCamFactor  
  int num_reprojection_one_frame_factors = 0; // ProjectionOneFrameTwoCamFactor
  int num_depth_prior_factors = 0;
  
  // Residual dimensions (total)
  int total_marginalization_residuals = 0;
  int total_imu_residuals = 0;
  int total_visual_residuals = 0;
  int total_depth_residuals = 0;
  
  // Feature statistics
  int num_features_in_optimization = 0;
  int num_features_tracked_long = 0;  // tracked >= 4 frames
  
  // Store residual block IDs for per-group cost evaluation
  std::vector<ceres::ResidualBlockId> marginalization_block_ids;
  std::vector<ceres::ResidualBlockId> imu_block_ids;
  std::vector<ceres::ResidualBlockId> visual_block_ids;
  std::vector<ceres::ResidualBlockId> depth_block_ids;
  
  // Per-feature tracking for debugging spikes
  std::vector<FeatureCostInfo> feature_costs;
  double (*para_Feature_ptr)[1] = nullptr;  // Pointer to para_Feature array for depth access
  
  // Per-group costs (before and after)
  double marginalization_cost_initial = 0.0;
  double marginalization_cost_final = 0.0;
  double imu_cost_initial = 0.0;
  double imu_cost_final = 0.0;
  double visual_cost_initial = 0.0;
  double visual_cost_final = 0.0;
  double depth_cost_initial = 0.0;
  double depth_cost_final = 0.0;
  
  // Add a new feature for per-feature cost tracking
  void addFeature(int feature_id, int feature_index, int start_frame, int num_obs, double inv_depth) {
    FeatureCostInfo info;
    info.feature_id = feature_id;
    info.feature_index = feature_index;
    info.start_frame = start_frame;
    info.num_observations = num_obs;
    info.inv_depth_before = inv_depth;
    info.inv_depth_after = inv_depth;
    info.cost_before = 0.0;
    info.cost_after = 0.0;
    feature_costs.push_back(info);
  }
  
  // Add residual block to the last added feature
  void addBlockToCurrentFeature(ceres::ResidualBlockId block_id) {
    if (!feature_costs.empty()) {
      feature_costs.back().block_ids.push_back(block_id);
    }
  }
  
  // Evaluate per-feature costs
  void evaluatePerFeatureCosts(ceres::Problem& problem, bool is_initial) {
    for (auto& fc : feature_costs) {
      if (fc.block_ids.empty()) continue;
      double cost = evaluateGroupCost(problem, fc.block_ids);
      if (is_initial) {
        fc.cost_before = cost;
      } else {
        fc.cost_after = cost;
        // Update final inverse depth
        if (para_Feature_ptr && fc.feature_index >= 0) {
          fc.inv_depth_after = para_Feature_ptr[fc.feature_index][0];
        }
      }
    }
  }
  
  // Print worst features (for debugging spikes)
  void printWorstFeatures(int top_n = 10) const {
    if (feature_costs.empty()) return;
    
    // Create sorted copy by initial cost
    std::vector<FeatureCostInfo> sorted_costs = feature_costs;
    std::sort(sorted_costs.begin(), sorted_costs.end(), 
              [](const FeatureCostInfo& a, const FeatureCostInfo& b) {
                return a.cost_before > b.cost_before;
              });
    
    double total_init_cost = 0.0;
    for (const auto& fc : feature_costs) total_init_cost += fc.cost_before;
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    TOP %d WORST FEATURES (by initial cost)                            ║\n", top_n);
    printf("╠═══════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Rank │ FeatureID │ StartFr │ #Obs │ InvD_Init │ InvD_Final │ Cost_Init  │ Cost_Final  ║\n");
    printf("╠──────┼───────────┼─────────┼──────┼───────────┼────────────┼────────────┼─────────────╣\n");
    
    int count = 0;
    double top_cost_sum = 0.0;
    for (const auto& fc : sorted_costs) {
      if (count >= top_n) break;
      top_cost_sum += fc.cost_before;
      printf("║ %4d │ %9d │ %7d │ %4d │ %9.4f │ %10.4f │ %10.2f │ %11.2f ║\n",
             count + 1, fc.feature_id, fc.start_frame, fc.num_observations,
             fc.inv_depth_before, fc.inv_depth_after, fc.cost_before, fc.cost_after);
      count++;
    }
    printf("╠═══════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Top %d features account for %.1f%% of total initial visual cost (%.2f / %.2f)        ║\n",
           top_n, 100.0 * top_cost_sum / (total_init_cost + 1e-10), top_cost_sum, total_init_cost);
    printf("╚═══════════════════════════════════════════════════════════════════════════════════════╝\n\n");
  }
  
  // Check for anomaly and print detailed debug info
  void checkForAnomaly(int frame_id) const {
    double total_init = visual_cost_initial + imu_cost_initial + marginalization_cost_initial;
    
    // Detect anomaly: visual cost > 10000 or visual cost > 95% of total
    bool is_anomaly = (visual_cost_initial > 10000.0) || 
                      (visual_cost_initial > 0.95 * total_init && total_init > 1000.0);
    
    if (is_anomaly) {
      printf("\n");
      printf("╔═══════════════════════════════════════════════════════════════════════════════════════╗\n");
      printf("║ ⚠️  ANOMALY DETECTED AT FRAME %d ⚠️                                                    ║\n", frame_id);
      printf("╠═══════════════════════════════════════════════════════════════════════════════════════╣\n");
      printf("║ Visual cost (init): %.2e - This is %.1fx higher than typical (~250)                  ║\n",
             visual_cost_initial, visual_cost_initial / 250.0);
      printf("║ Marginalization cost increased: %.2f → %.2f (%.1f%%)                                 ║\n",
             marginalization_cost_initial, marginalization_cost_final,
             100.0 * (marginalization_cost_final - marginalization_cost_initial) / (marginalization_cost_initial + 1e-10));
      printf("║ IMU cost change: %.2f → %.2f (%.1f%%)                                                ║\n",
             imu_cost_initial, imu_cost_final,
             100.0 * (imu_cost_final - imu_cost_initial) / (imu_cost_initial + 1e-10));
      printf("╚═══════════════════════════════════════════════════════════════════════════════════════╝\n");
      
      // Print worst features when anomaly detected
      printWorstFeatures(15);
    }
  }
  
  // Helper function to evaluate cost for a subset of residual blocks
  double evaluateGroupCost(ceres::Problem& problem, 
                          const std::vector<ceres::ResidualBlockId>& block_ids) {
    if (block_ids.empty()) return 0.0;
    
    double total_cost = 0.0;
    ceres::Problem::EvaluateOptions eval_options;
    eval_options.residual_blocks = block_ids;
    eval_options.apply_loss_function = true;
    
    problem.Evaluate(eval_options, &total_cost, nullptr, nullptr, nullptr);
    return total_cost;
  }
  
  void evaluateInitialCosts(ceres::Problem& problem) {
    marginalization_cost_initial = evaluateGroupCost(problem, marginalization_block_ids);
    imu_cost_initial = evaluateGroupCost(problem, imu_block_ids);
    visual_cost_initial = evaluateGroupCost(problem, visual_block_ids);
    depth_cost_initial = evaluateGroupCost(problem, depth_block_ids);
  }
  
  void evaluateFinalCosts(ceres::Problem& problem) {
    marginalization_cost_final = evaluateGroupCost(problem, marginalization_block_ids);
    imu_cost_final = evaluateGroupCost(problem, imu_block_ids);
    visual_cost_final = evaluateGroupCost(problem, visual_block_ids);
    depth_cost_final = evaluateGroupCost(problem, depth_block_ids);
  }
  
  void print() const {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║           OPTIMIZATION FACTOR DIAGNOSTICS                     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║ Factor Type                    │ Count   │ Residual Dim       ║\n");
    printf("╠────────────────────────────────┼─────────┼────────────────────╣\n");
    printf("║ Marginalization                │ %7d │ %7d            ║\n", 
           num_marginalization_factors, total_marginalization_residuals);
    printf("║ IMU Preintegration             │ %7d │ %7d (15 each)   ║\n", 
           num_imu_factors, total_imu_residuals);
    printf("║ Reprojection (Mono 2-Frame)    │ %7d │ %7d (2 each)    ║\n", 
           num_reprojection_mono_factors, num_reprojection_mono_factors * 2);
    printf("║ Reprojection (Stereo 2-Frame)  │ %7d │ %7d (2 each)    ║\n", 
           num_reprojection_stereo_factors, num_reprojection_stereo_factors * 2);
    printf("║ Reprojection (Stereo 1-Frame)  │ %7d │ %7d (2 each)    ║\n", 
           num_reprojection_one_frame_factors, num_reprojection_one_frame_factors * 2);
    printf("║ Depth Prior                    │ %7d │ %7d (1 each)    ║\n", 
           num_depth_prior_factors, total_depth_residuals);
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    
    int total_factors = num_marginalization_factors + num_imu_factors + 
                       num_reprojection_mono_factors + num_reprojection_stereo_factors +
                       num_reprojection_one_frame_factors + num_depth_prior_factors;
    int total_visual = num_reprojection_mono_factors + num_reprojection_stereo_factors + 
                      num_reprojection_one_frame_factors;
    int total_residuals = total_marginalization_residuals + total_imu_residuals +
                         total_visual * 2 + total_depth_residuals;
    
    printf("║ TOTALS                                                        ║\n");
    printf("║   Total factors:        %7d                                ║\n", total_factors);
    printf("║   Total visual factors: %7d                                ║\n", total_visual);
    printf("║   Total residual dim:   %7d                                ║\n", total_residuals);
    printf("║   Features optimized:   %7d                                ║\n", num_features_in_optimization);
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
  }
  
  void printCostBreakdown() const {
    double total_initial = marginalization_cost_initial + imu_cost_initial + 
                          visual_cost_initial + depth_cost_initial;
    double total_final = marginalization_cost_final + imu_cost_final + 
                        visual_cost_final + depth_cost_final;
    
    printf("\n");
    printf("╔═════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    PER-FACTOR-GROUP COST BREAKDOWN                              ║\n");
    printf("╠═════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Factor Group          │ Initial Cost   │ Final Cost     │ Reduction    │ %%     ║\n");
    printf("╠───────────────────────┼────────────────┼────────────────┼──────────────┼───────╣\n");
    
    auto printRow = [](const char* name, double initial, double final_cost) {
      double reduction = initial - final_cost;
      double percent = (initial > 1e-10) ? 100.0 * reduction / initial : 0.0;
      printf("║ %-21s │ %14.4e │ %14.4e │ %12.4e │ %5.1f%% ║\n",
             name, initial, final_cost, reduction, percent);
    };
    
    printRow("Marginalization", marginalization_cost_initial, marginalization_cost_final);
    printRow("IMU Preintegration", imu_cost_initial, imu_cost_final);
    printRow("Visual Reprojection", visual_cost_initial, visual_cost_final);
    printRow("Depth Prior", depth_cost_initial, depth_cost_final);
    
    printf("╠───────────────────────┼────────────────┼────────────────┼──────────────┼───────╣\n");
    printRow("TOTAL", total_initial, total_final);
    printf("╠═════════════════════════════════════════════════════════════════════════════════╣\n");
    
    // Print percentage contribution of each factor group
    printf("║ Cost Distribution (Initial):                                                    ║\n");
    if (total_initial > 1e-10) {
      printf("║   Marginalization: %5.1f%% | IMU: %5.1f%% | Visual: %5.1f%% | Depth: %5.1f%%        ║\n",
             100.0 * marginalization_cost_initial / total_initial,
             100.0 * imu_cost_initial / total_initial,
             100.0 * visual_cost_initial / total_initial,
             100.0 * depth_cost_initial / total_initial);
    }
    printf("║ Cost Distribution (Final):                                                      ║\n");
    if (total_final > 1e-10) {
      printf("║   Marginalization: %5.1f%% | IMU: %5.1f%% | Visual: %5.1f%% | Depth: %5.1f%%        ║\n",
             100.0 * marginalization_cost_final / total_final,
             100.0 * imu_cost_final / total_final,
             100.0 * visual_cost_final / total_final,
             100.0 * depth_cost_final / total_final);
    }
    printf("╚═════════════════════════════════════════════════════════════════════════════════╝\n\n");
  }
  
  // CSV export for later visualization
  void exportToCSV(const std::string& filepath, double timestamp, int frame_id, 
                   double solver_time_ms, int iterations, const std::string& termination) {
    static bool header_written = false;
    
    std::ofstream file;
    if (!header_written) {
      // Check if file exists and has content
      std::ifstream check_file(filepath);
      if (check_file.good() && check_file.peek() != std::ifstream::traits_type::eof()) {
        header_written = true;  // File exists with content, don't write header
      }
      check_file.close();
    }
    
    file.open(filepath, std::ios::app);
    if (!file.is_open()) {
      std::cerr << "[OptDiag] Failed to open CSV file: " << filepath << std::endl;
      return;
    }
    
    // Write header if this is the first write
    if (!header_written) {
      file << "timestamp,frame_id,solver_time_ms,iterations,termination,"
           << "num_margin_factors,num_imu_factors,num_visual_mono_factors,"
           << "num_visual_stereo_factors,num_visual_one_frame_factors,num_depth_factors,"
           << "num_features,total_residual_dim,"
           << "margin_cost_init,margin_cost_final,"
           << "imu_cost_init,imu_cost_final,"
           << "visual_cost_init,visual_cost_final,"
           << "depth_cost_init,depth_cost_final,"
           << "total_cost_init,total_cost_final,"
           << "margin_pct_init,imu_pct_init,visual_pct_init,depth_pct_init,"
           << "margin_pct_final,imu_pct_final,visual_pct_final,depth_pct_final,"
           << "margin_reduction_pct,imu_reduction_pct,visual_reduction_pct,depth_reduction_pct,"
           << "total_reduction_pct\n";
      header_written = true;
    }
    
    // Compute derived values
    double total_initial = marginalization_cost_initial + imu_cost_initial + 
                          visual_cost_initial + depth_cost_initial;
    double total_final = marginalization_cost_final + imu_cost_final + 
                        visual_cost_final + depth_cost_final;
    
    int total_visual = num_reprojection_mono_factors + num_reprojection_stereo_factors + 
                      num_reprojection_one_frame_factors;
    int total_residuals = total_marginalization_residuals + total_imu_residuals +
                         total_visual * 2 + total_depth_residuals;
    
    auto safePct = [](double part, double total) -> double {
      return (total > 1e-10) ? 100.0 * part / total : 0.0;
    };
    
    auto safeReduction = [](double init, double final_val) -> double {
      return (init > 1e-10) ? 100.0 * (init - final_val) / init : 0.0;
    };
    
    // Write data row
    file << std::fixed << std::setprecision(6) << timestamp << ","
         << frame_id << ","
         << std::setprecision(2) << solver_time_ms << ","
         << iterations << ","
         << termination << ","
         << num_marginalization_factors << ","
         << num_imu_factors << ","
         << num_reprojection_mono_factors << ","
         << num_reprojection_stereo_factors << ","
         << num_reprojection_one_frame_factors << ","
         << num_depth_prior_factors << ","
         << num_features_in_optimization << ","
         << total_residuals << ","
         << std::scientific << std::setprecision(6)
         << marginalization_cost_initial << ","
         << marginalization_cost_final << ","
         << imu_cost_initial << ","
         << imu_cost_final << ","
         << visual_cost_initial << ","
         << visual_cost_final << ","
         << depth_cost_initial << ","
         << depth_cost_final << ","
         << total_initial << ","
         << total_final << ","
         << std::fixed << std::setprecision(2)
         << safePct(marginalization_cost_initial, total_initial) << ","
         << safePct(imu_cost_initial, total_initial) << ","
         << safePct(visual_cost_initial, total_initial) << ","
         << safePct(depth_cost_initial, total_initial) << ","
         << safePct(marginalization_cost_final, total_final) << ","
         << safePct(imu_cost_final, total_final) << ","
         << safePct(visual_cost_final, total_final) << ","
         << safePct(depth_cost_final, total_final) << ","
         << safeReduction(marginalization_cost_initial, marginalization_cost_final) << ","
         << safeReduction(imu_cost_initial, imu_cost_final) << ","
         << safeReduction(visual_cost_initial, visual_cost_final) << ","
         << safeReduction(depth_cost_initial, depth_cost_final) << ","
         << safeReduction(total_initial, total_final) << "\n";
    
    file.close();
  }
};

void Estimator::optimization() {
  std::cout << "\n========== DEBUG: Optimization all_image_frame STATUS ==========" << std::endl;
  std::cout << "Current Window Size: " << frame_count << std::endl;
  
  // Initialize diagnostics tracker
  OptimizationDiagnostics diag;
  
  int added_depth_factors = 0;
  int idx = 0;
  for (const auto& item : all_image_frame) {
      double t = item.first;
      bool has_depth = !item.second.depth_map.empty();
      
      //std::cout << "Frame[" << idx++ << "] TS: " 
      //          << std::fixed << std::setprecision(9) << t;
      //
      //if (has_depth) {
      //    std::cout << " | [HAS DEPTH] " << item.second.depth_map.cols << "x" << item.second.depth_map.rows;
      //} else {
      //    std::cout << " | [NO DEPTH]";
      //}
      //std::cout << std::endl;
  }
  
  //std::cout << "-------- CURRENT WINDOW HEADERS (For Comparison) --------" << std::endl;
  //for (int i = 0; i <= frame_count; i++) {
  //    double t = Headers[i];
  //    bool is_key = false;
  //    if (all_image_frame.find(t) != all_image_frame.end()) {
  //        is_key = all_image_frame[t].is_optimization_keyframe;
  //    }
  //    std::cout << "Header[" << i << "]: " 
  //              << std::fixed << std::setprecision(6) << t << (is_key ? " KF" : " not KF") << std::endl;
  //}
  //std::cout << "=========================================================\n" << std::endl;

  TicToc t_whole;
  TicToc t_prepare;
  vector2double();
  int qualified = 0;
  
  ceres::Problem problem;
  ceres::LossFunction *loss_function;

  if (params.loss_type == LOSS_HUBER) {
    loss_function = new ceres::HuberLoss(params.loss_parameter);
  } else if (params.loss_type == LOSS_CAUCHY) {
    loss_function = new ceres::CauchyLoss(params.loss_parameter);
  } else if (params.loss_type == LOSS_TUKEY) {
    loss_function = new ceres::TukeyLoss(params.loss_parameter);
  } else if (params.loss_type == LOSS_L2) {
    loss_function = nullptr;
  } else {
    ROS_ERROR("Unknown loss type!");
    exit(-1);
  }

  for (int i = 0; i < frame_count + 1; i++) {
    ceres::Manifold *manifold = new PoseManifold();
    problem.AddParameterBlock(para_Pose[i], SIZE_POSE);
    problem.SetManifold(para_Pose[i], manifold);
    if (params.use_imu)
      problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
  }
  if (!params.use_imu) problem.SetParameterBlockConstant(para_Pose[0]);

  for (int i = 0; i < params.num_of_cam; i++) {
    ceres::Manifold *manifold = new PoseManifold();
    problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE);
    problem.SetManifold(para_Ex_Pose[i], manifold);
    if ((params.estimate_extrinsic && frame_count == WINDOW_SIZE &&
         Vs[0].norm() > 0.2) ||
        openExEstimation) {
      // ROS_INFO("estimate extinsic param");
      openExEstimation = true;
    } else {
      // ROS_INFO("fix extinsic param");
      problem.SetParameterBlockConstant(para_Ex_Pose[i]);
    }
  }
  problem.AddParameterBlock(para_Td[0], 1);

  if (!params.estimate_td || Vs[0].norm() < 0.2)
    problem.SetParameterBlockConstant(para_Td[0]);

  if (last_marginalization_info && last_marginalization_info->valid) {
    // construct new marginlization_factor
    MarginalizationFactor *marginalization_factor =
        new MarginalizationFactor(last_marginalization_info);
    ceres::ResidualBlockId block_id = problem.AddResidualBlock(marginalization_factor, NULL,
                             last_marginalization_parameter_blocks);
    diag.marginalization_block_ids.push_back(block_id);
    diag.num_marginalization_factors++;
    diag.total_marginalization_residuals = last_marginalization_info->n;  // marginalization residual dimension
  }
  if (params.use_imu) {
    for (int i = 0; i < frame_count; i++) {
      int j = i + 1;
      if (pre_integrations[j]->sum_dt > 10.0) continue;
      IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
      ceres::ResidualBlockId block_id = problem.AddResidualBlock(imu_factor, NULL, para_Pose[i],
                               para_SpeedBias[i], para_Pose[j],
                               para_SpeedBias[j]);
      diag.imu_block_ids.push_back(block_id);
      diag.num_imu_factors++;
      diag.total_imu_residuals += 15;  // IMU factor has 15 residuals (3 pos + 3 vel + 3 rot + 3 ba + 3 bg)
    }
  }

  int f_m_cnt = 0;
  int feature_index = -1;
  ceres::LossFunction *depth_align_loss_function = new ceres::HuberLoss(0.1);
  // Just to use for debug later
  for (auto &it_per_id : f_manager.feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4) continue;
    int first_frame_idx = it_per_id.start_frame;
    numbers[first_frame_idx]++;
    diag.num_features_tracked_long++;
  }
  int skipped_outliers = 0;  // Count features skipped due to high initial error
  
  double t_temporal_cost = 0.0;
  for (auto &it_per_id : f_manager.feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4) continue;
    ++feature_index;
    
    // =========================================================================
    // PRE-OPTIMIZATION OUTLIER CHECK: Skip features with huge reprojection error
    // This prevents single bad features from corrupting the entire optimization
    // Toggle via config: preopt_outlier_filter (bool)
    // =========================================================================
    if (params.preopt_outlier_filter) {
      double depth = 1.0 / para_Feature[feature_index][0];  // Convert inv_depth to depth
      
      // Skip features with invalid depth (negative or extremely large)
      if (depth <= 0.1 || depth > 200.0) {
        skipped_outliers++;
        continue;
      }
      
      // Skip features at image edges - they have poor tracking and distortion
      Vector3d pts_i_edge_check = it_per_id.feature_per_frame[0].point;
      if (std::abs(pts_i_edge_check.x()) > params.preopt_edge_threshold || 
          std::abs(pts_i_edge_check.y()) > params.preopt_edge_threshold) {
        // printf("\033[1;36m[EDGE] Feature %d skipped: pts=(%.3f, %.3f) near image edge\033[0m\n",
        //        it_per_id.feature_id, pts_i_edge_check.x(), pts_i_edge_check.y());
        skipped_outliers++;
        continue;
      }
      
      // Compute average reprojection error across all observations
      double total_reproj_error = 0.0;
      double max_reproj_error = 0.0;
      int error_cnt = 0;
      int imu_i_check = it_per_id.start_frame;
      int imu_j_check = imu_i_check - 1;
      Vector3d pts_i_check = it_per_id.feature_per_frame[0].point;
      
      // Store individual errors for debugging
      std::vector<std::tuple<int, double, Vector3d, Vector3d>> per_obs_errors;  // (frame_j, error, pts_i, pts_j)
      
    
      for (auto &it_per_frame : it_per_id.feature_per_frame) {
        imu_j_check++;
        if (imu_i_check != imu_j_check) {
          Vector3d pts_j_check = it_per_frame.point;
          double err = reprojectionError(Rs[imu_i_check], Ps[imu_i_check], ric[0], tic[0],
                                         Rs[imu_j_check], Ps[imu_j_check], ric[0], tic[0],
                                         depth, pts_i_check, pts_j_check);
          total_reproj_error += err;
          max_reproj_error = std::max(max_reproj_error, err);
          error_cnt++;
          per_obs_errors.push_back({imu_j_check, err, pts_i_check, pts_j_check});
        }
      }
      
      
      // Skip feature if average reprojection error is too high
      double avg_reproj_error = (error_cnt > 0) ? total_reproj_error / error_cnt : 0.0;
      
      if (avg_reproj_error > params.preopt_reproj_error_threshold) {
        printf("\033[1;33m[PRE-OPT OUTLIER] Feature %d skipped: avg_reproj_err=%.4f, max=%.4f, depth=%.2f, start_frame=%d, #obs=%d\033[0m\n",
               it_per_id.feature_id, avg_reproj_error, max_reproj_error, depth, it_per_id.start_frame, (int)it_per_id.feature_per_frame.size());
        
        // Print detailed per-observation info for very bad features
        if (avg_reproj_error > 5.0) {
          printf("    \033[1;31m[SEVERE] pts_i (normalized): (%.4f, %.4f, %.4f)\033[0m\n", 
                 pts_i_check.x(), pts_i_check.y(), pts_i_check.z());
          printf("    \033[1;31m[SEVERE] Pose_i: pos=(%.2f, %.2f, %.2f)\033[0m\n",
                 Ps[imu_i_check].x(), Ps[imu_i_check].y(), Ps[imu_i_check].z());
          for (const auto& obs : per_obs_errors) {
            int fr = std::get<0>(obs);
            double e = std::get<1>(obs);
            Vector3d pj = std::get<3>(obs);
            printf("    -> Frame %d: err=%.4f, pts_j=(%.4f, %.4f), Pose_j=(%.2f, %.2f, %.2f)\n",
                   fr, e, pj.x(), pj.y(), Ps[fr].x(), Ps[fr].y(), Ps[fr].z());
          }
        }
        
        skipped_outliers++;
        continue;
      }
    }  // end preopt_outlier_filter
    
    diag.num_features_in_optimization++;
    
    // Track this feature for per-feature cost analysis
    diag.addFeature(it_per_id.feature_id, feature_index, it_per_id.start_frame, 
                    it_per_id.used_num, para_Feature[feature_index][0]);

    int imu_i = it_per_id.start_frame;
    int imu_j = imu_i - 1;

    Vector3d pts_i = it_per_id.feature_per_frame[0].point;

    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      if (imu_i != imu_j) {
        Vector3d pts_j = it_per_frame.point;
        ProjectionTwoFrameOneCamFactor *f_td =
            new ProjectionTwoFrameOneCamFactor(
                pts_i, pts_j, it_per_id.feature_per_frame[0].velocity,
                it_per_frame.velocity, it_per_id.feature_per_frame[0].cur_td,
                it_per_frame.cur_td);
        ceres::ResidualBlockId block_id = problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i],
                                 para_Pose[imu_j], para_Ex_Pose[0],
                                 para_Feature[feature_index], para_Td[0]);
        diag.visual_block_ids.push_back(block_id);
        diag.addBlockToCurrentFeature(block_id);  // Track per-feature
        diag.num_reprojection_mono_factors++;
      }

      if (params.stereo && it_per_frame.is_stereo) {
        Vector3d pts_j_right = it_per_frame.pointRight;
        if (imu_i != imu_j) {
          ProjectionTwoFrameTwoCamFactor *f =
              new ProjectionTwoFrameTwoCamFactor(
                  pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity,
                  it_per_frame.velocityRight,
                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
          ceres::ResidualBlockId block_id = problem.AddResidualBlock(f, loss_function, para_Pose[imu_i],
                                   para_Pose[imu_j], para_Ex_Pose[0],
                                   para_Ex_Pose[1], para_Feature[feature_index],
                                   para_Td[0]);
          diag.visual_block_ids.push_back(block_id);
          diag.addBlockToCurrentFeature(block_id);  // Track per-feature
          diag.num_reprojection_stereo_factors++;
        } else {
          ProjectionOneFrameTwoCamFactor *f =
              new ProjectionOneFrameTwoCamFactor(
                  pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity,
                  it_per_frame.velocityRight,
                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
          ceres::ResidualBlockId block_id = problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0],
                                   para_Ex_Pose[1], para_Feature[feature_index],
                                   para_Td[0]);
          diag.visual_block_ids.push_back(block_id);
          diag.addBlockToCurrentFeature(block_id);  // Track per-feature
          diag.num_reprojection_one_frame_factors++;
        }
      }
      f_m_cnt++;
    }
    
    // =====================================================================
    // DEPTH PRIOR FACTOR (Using pre-aligned global scale/shift from RANSAC)
    // =====================================================================
    // Key insight: Instead of optimizing scale/shift per-frame (too many unknowns),
    // we use the globally learned scale/shift from smartDepthInitialization()
    // and add a simple prior pulling VIO depths toward aligned mono depths.

    if (solver_flag == NON_LINEAR && params.use_depth && WEIGHT > 0.0 && scale_is_initialized) {
      int first_frame_idx = it_per_id.start_frame;
      
      // Skip if not enough features for reliable alignment
      //if (numbers[first_frame_idx] < params.min_features) continue;
      
      // Skip early frames to let triangulation settle
      if (inputImageCnt < 150) continue;
      
      // Skip features too close to window edge
      if (first_frame_idx > WINDOW_SIZE - 2) continue;
      
      // Skip features that haven't been tracked long enough
      if (it_per_id.feature_per_frame.size() < 3) continue;
      
      double timestamp = Headers[first_frame_idx];
      auto frame_it = all_image_frame.find(timestamp);
      
      if (frame_it != all_image_frame.end() && !frame_it->second.depth_map.empty()) {
        const cv::Mat& depth_map = frame_it->second.depth_map;
        auto feature_data_it = frame_it->second.points.find(it_per_id.feature_id);
        
        if (feature_data_it != frame_it->second.points.end()) {
          const auto& measurement = feature_data_it->second[0].second;
          int x_px = static_cast<int>(measurement(3));
          int y_px = static_cast<int>(measurement(4));
          
          if (x_px >= 2 && x_px < depth_map.cols - 2 && 
              y_px >= 2 && y_px < depth_map.rows - 2) {
            
            // --- ROBUST DEPTH SAMPLING ---
            // Collect 3x3 patch and use median (robust to edge effects)
            std::vector<float> patch_vals;
            patch_vals.reserve(9);
            for (int dy = -1; dy <= 1; dy++) {
              for (int dx = -1; dx <= 1; dx++) {
                float val = depth_map.at<float>(y_px + dy, x_px + dx);
                if (val > 0.001f) patch_vals.push_back(val);
              }
            }
            
            if (patch_vals.size() >= 5) {  // Need enough valid samples
              // Median is more robust than max at depth discontinuities
              std::nth_element(patch_vals.begin(), 
                              patch_vals.begin() + patch_vals.size()/2, 
                              patch_vals.end());
              float mono_inv_depth = patch_vals[patch_vals.size()/2];
              
              // Check depth consistency (reject if patch variance is too high)
              float min_val = *std::min_element(patch_vals.begin(), patch_vals.end());
              float max_val = *std::max_element(patch_vals.begin(), patch_vals.end());
              float depth_consistency = min_val / (max_val + 1e-6f);
              
              if (depth_consistency > 0.7f) {  // Patch is relatively uniform
                // Apply pre-computed global alignment
                double aligned_inv_depth = cached_scale * mono_inv_depth + cached_shift;
                
                // ================================================================
                // TEMPORAL STABILITY: Update depth history and check variance
                // Only enabled if params.temporal_stable == 1
                // ================================================================
                if (params.temporal_stable) {
                  TicToc t_temp;
                  // Update the feature's depth history with this aligned measurement
                  it_per_id.updateDepthHistory(aligned_inv_depth, 
                                                params.temporal_stable_buffer_size,
                                                params.temporal_stable_variance_thresh);
                  t_temporal_cost += t_temp.toc();
                }
                
                // Sanity checks
                float vins_inv_depth = para_Feature[feature_index][0];
                float vins_metric_depth = 1.0f / vins_inv_depth;
                
                if (aligned_inv_depth > 0.01 && aligned_inv_depth < 10.0 &&  // Valid inv depth range
                    vins_metric_depth > 0.1 && vins_metric_depth < 100.0) {  // VIO depth reasonable
                  
                  // ================================================================
                  // TEMPORAL STABILITY CHECK: Skip or downweight unstable features
                  // ================================================================
                  bool should_add_prior = true;
                  double temporal_weight_factor = 1.0;
                  
                  if (params.temporal_stable) {
                    if (!it_per_id.depth_stable) {
                      // Feature has flickering depth - skip adding absolute prior
                      // It will still be eligible for ordinal constraints
                      should_add_prior = false;
                    } else {
                      // Stable feature - optionally boost weight based on low variance
                      // Lower variance = more confidence = higher weight
                      double var_ratio = it_per_id.depth_variance / (params.temporal_stable_variance_thresh + 1e-8);
                      temporal_weight_factor = std::max(0.5, 1.0 - var_ratio);  // Range [0.5, 1.0]
                    }
                  }
                  
                  if (should_add_prior) {
                    double adaptive_weight = WEIGHT * temporal_weight_factor;
                    
                    // ================================================================
                    // MAHALANOBIS DISTANCE-BASED WEIGHTING (Inverse Domain)
                    // Only enabled if use_mahalanobis_weight == 1 in config
                    // ================================================================
                    if (params.use_mahalanobis_weight) {
                      // Compute the discrepancy between VIO and aligned depth in inverse domain
                      double inv_depth_error = vins_inv_depth - aligned_inv_depth;
                      
                      // Mahalanobis distance: d_M = |error - mean| / sqrt(variance)
                      // This measures how many standard deviations away this measurement is
                      double std_dev = std::sqrt(cached_inv_depth_variance);
                      double mahalanobis_dist = std::abs(inv_depth_error - cached_inv_depth_mean_error) / (std_dev + 1e-8);
                      
                      // Adaptive weight based on Mahalanobis distance
                      // Features with large discrepancy (outliers) get lower weight
                      // Using a soft thresholding function: w = base_weight * exp(-k * d_M^2)
                      // This gives:
                      //   - Full weight when d_M ≈ 0 (measurement agrees with model)
                      //   - Exponentially decreasing weight for outliers
                      //   - k controls how quickly weight drops (k=0.5 means ~60% weight at 1 std dev)
                      double k_mahal = 0.5;  // Tuning parameter for weight falloff
                      double mahal_weight_factor = std::exp(-k_mahal * mahalanobis_dist * mahalanobis_dist);
                      
                      // Clamp minimum weight to avoid completely ignoring any measurement
                      mahal_weight_factor = std::max(mahal_weight_factor, 0.1);
                      
                      adaptive_weight *= mahal_weight_factor;
                    }
                    
                    // Add the depth prior factor with (optionally Mahalanobis-weighted) information
                    // Pass residual_log flag to control log vs linear residual
                    bool use_log_residual = (params.residual_log == 1);
                    ceres::CostFunction* cost_function = 
                        DepthPriorFactor::Create(aligned_inv_depth, adaptive_weight, use_log_residual);
                    ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function,
                                            depth_align_loss_function,
                                            para_Feature[feature_index]);
                    diag.depth_block_ids.push_back(block_id);
                    added_depth_factors++;
                    diag.num_depth_prior_factors++;
                    diag.total_depth_residuals++;
                  }
                }
              }
            }
          }
        }
      }
    }
  }  // end feature loop

  if (params.temporal_stable && t_temporal_cost > 0.0) {
      ROS_INFO("[Temporal] Time cost: %f ms", t_temporal_cost);
  }
  
  // Report skipped outliers
  if (skipped_outliers > 0) {
    printf("\033[1;35m[PRE-OPT] Skipped %d features with high initial reprojection error\033[0m\n", 
           skipped_outliers);
  }
  
  // =====================================================================
  // ORDINAL DEPTH CONSTRAINTS (Relative depth ordering)
  // Only enabled if params.ordinal_depth == 1
  // =====================================================================
  int added_ordinal_factors = 0;
  
  if (solver_flag == NON_LINEAR && params.ordinal_depth && params.use_depth && scale_is_initialized) {
    TicToc t_ordinal;
    // Collect features with valid depth measurements for ordinal pairing
    // Structure: {feature_index, mono_inv_depth, frame_idx}
    struct OrdinalCandidate {
      int feature_index;
      double mono_inv_depth;
      int frame_idx;
      int pixel_x, pixel_y;  // For spatial proximity check
    };
    
    std::vector<OrdinalCandidate> candidates;
    candidates.reserve(200);
    
    // Re-iterate features to collect ordinal candidates
    // (We do this separately to avoid complicating the main feature loop)
    int ordinal_feature_idx = -1;
    for (auto &it_per_id : f_manager.feature) {
      it_per_id.used_num = it_per_id.feature_per_frame.size();
      if (it_per_id.used_num < 4) continue;
      ++ordinal_feature_idx;
      
      int first_frame_idx = it_per_id.start_frame;
      if (first_frame_idx > WINDOW_SIZE - 2) continue;
      
      double timestamp = Headers[first_frame_idx];
      auto frame_it = all_image_frame.find(timestamp);
      
      if (frame_it != all_image_frame.end() && !frame_it->second.depth_map.empty()) {
        const cv::Mat& depth_map = frame_it->second.depth_map;
        auto feature_data_it = frame_it->second.points.find(it_per_id.feature_id);
        
        if (feature_data_it != frame_it->second.points.end()) {
          const auto& measurement = feature_data_it->second[0].second;
          int x_px = static_cast<int>(measurement(3));
          int y_px = static_cast<int>(measurement(4));
          
          if (x_px >= 1 && x_px < depth_map.cols - 1 && 
              y_px >= 1 && y_px < depth_map.rows - 1) {
            float mono_inv_depth = depth_map.at<float>(y_px, x_px);
            
            if (mono_inv_depth > 0.001f) {
              candidates.push_back({ordinal_feature_idx, mono_inv_depth, first_frame_idx, x_px, y_px});
            }
          }
        }
      }
    }
    
    // Generate ordinal pairs: features in same frame with significant depth difference
    std::vector<OrdinalPair> ordinal_pairs;
    ordinal_pairs.reserve(params.ordinal_depth_max_pairs);
    
    for (size_t i = 0; i < candidates.size() && ordinal_pairs.size() < static_cast<size_t>(params.ordinal_depth_max_pairs); i++) {
      for (size_t j = i + 1; j < candidates.size() && ordinal_pairs.size() < static_cast<size_t>(params.ordinal_depth_max_pairs); j++) {
        // Only pair features from the same frame
        if (candidates[i].frame_idx != candidates[j].frame_idx) continue;
        
        double diff = candidates[i].mono_inv_depth - candidates[j].mono_inv_depth;
        double abs_diff = std::abs(diff);
        
        // Only create constraint if depth difference is significant
        // (avoid constraining features at similar depths)
        if (abs_diff > 0.05) {  // Significant inv-depth difference
          OrdinalPair pair;
          if (diff > 0) {
            // i is closer (higher inv-depth)
            pair.feature_idx_closer = candidates[i].feature_index;
            pair.feature_idx_farther = candidates[j].feature_index;
          } else {
            // j is closer
            pair.feature_idx_closer = candidates[j].feature_index;
            pair.feature_idx_farther = candidates[i].feature_index;
          }
          pair.inv_depth_diff = abs_diff;
          ordinal_pairs.push_back(pair);
        }
      }
    }
    
    // Sort by confidence (larger depth difference = more confident ordering)
    std::sort(ordinal_pairs.begin(), ordinal_pairs.end(),
              [](const OrdinalPair& a, const OrdinalPair& b) {
                return a.inv_depth_diff > b.inv_depth_diff;
              });
    
    // Add ordinal constraints (limit to max_pairs)
    int pairs_to_add = std::min(static_cast<int>(ordinal_pairs.size()), params.ordinal_depth_max_pairs);
    for (int i = 0; i < pairs_to_add; i++) {
      const auto& pair = ordinal_pairs[i];
      
      ceres::CostFunction* cost_function = 
          OrdinalDepthFactor::Create(params.ordinal_depth_margin, params.ordinal_depth_weight);
      problem.AddResidualBlock(cost_function,
                               nullptr,  // No loss function - soft hinge built into factor
                               para_Feature[pair.feature_idx_closer],
                               para_Feature[pair.feature_idx_farther]);
      added_ordinal_factors++;
    }
    
    if (added_ordinal_factors > 0) {
      printf("\033[1;35m[Ordinal Depth] Added %d ordinal constraints (margin=%.3f, w=%.2f)\033[0m\n",
             added_ordinal_factors, params.ordinal_depth_margin, params.ordinal_depth_weight);
    }
    ROS_INFO("[Ordinal] Time cost: %f ms", t_ordinal.toc());
  }
  
  // Debug output
  if (added_depth_factors > 0) {
    printf("\033[1;36m[Depth Opt] Added %d depth priors (s=%.3f, t=%.3f, w=%.2f, var=%.6f)\033[0m\n", 
           added_depth_factors, cached_scale, cached_shift, WEIGHT, cached_inv_depth_variance);
  }
  added_depth_factors = 0; //Reset for next optimization call
  std::cout << "Feature count per frame: ";
  for (int i = 0; i <= WINDOW_SIZE; i++) {
      std::cout << numbers[i] << (i == WINDOW_SIZE ? "\n" : ", ");
      numbers[i] = 0; // Reset for next loop
  }

  // Print factor diagnostics BEFORE solving
  diag.print();
  
  // Store pointer to para_Feature for per-feature analysis
  diag.para_Feature_ptr = para_Feature;
  
  // Evaluate per-group costs BEFORE optimization
  diag.evaluateInitialCosts(problem);
  diag.evaluatePerFeatureCosts(problem, true);  // Per-feature costs (initial)

  ROS_DEBUG("visual measurement count: %d", f_m_cnt);
  // printf("prepare for ceres: %f \n", t_prepare.toc());

  ceres::Solver::Options options;

  //options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.num_threads = 2;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = params.num_iterations;
  // options.use_explicit_schur_complement = true;
  // options.minimizer_progress_to_stdout = true;
  // options.use_nonmonotonic_steps = true;
  if (params.use_cuda_in_optimization){
    #ifdef VINS_USE_CUDA 
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      options.dense_linear_algebra_library_type = ceres::CUDA;
      options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
      // Orin has many cores, let Ceres use them for Jacobian evaluation
      options.num_threads = 6; 
      ROS_WARN_ONCE("\033[1;32m[VINS-GPU] Ceres is compiled with CUDA support! Using GPU Solver.\033[0m");
      std::cout << "[VINS-GPU] Dense Algebra: CUDA" << std::endl;
      std::cout << "[VINS-GPU] Sparse Algebra: CUDA_SPARSE" << std::endl;
    #else
      // 3. DEBUG PRINT: Warn if we missed the definition
      ROS_WARN_ONCE("\033[1;31m[VINS-CPU] Ceres CUDA support NOT found. Falling back to CPU.\033[0m");
      // Standard CPU Fallback
      options.linear_solver_type = ceres::DENSE_SCHUR;
    #endif
  }
  else{
    options.linear_solver_type = ceres::DENSE_SCHUR;
  }
  if (marginalization_flag == MARGIN_OLD)
    options.max_solver_time_in_seconds = params.solver_time * 4.0 / 5.0;
  else
    options.max_solver_time_in_seconds = params.solver_time;
  TicToc t_solver;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  
  // Evaluate per-group costs AFTER optimization
  diag.evaluateFinalCosts(problem);
  diag.evaluatePerFeatureCosts(problem, false);  // Per-feature costs (final)
  
  // Check for anomalies and print detailed debug info if detected
  diag.checkForAnomaly(inputImageCnt);

  // Enhanced solver diagnostics
  printf("\n");
  printf("╔═══════════════════════════════════════════════════════════════╗\n");
  printf("║              CERES OPTIMIZATION RESULTS                       ║\n");
  printf("╠═══════════════════════════════════════════════════════════════╣\n");
  printf("║ Solver Time:          %10.2f ms                           ║\n", 
         summary.total_time_in_seconds * 1000.0);
  printf("║ Iterations:           %10d                              ║\n", 
         (int)summary.iterations.size());
  printf("║ Initial Cost:         %10.4e                           ║\n", 
         summary.initial_cost);
  printf("║ Final Cost:           %10.4e                           ║\n", 
         summary.final_cost);
  printf("║ Cost Change:          %10.4e (%.2f%%)                  ║\n", 
         summary.initial_cost - summary.final_cost,
         (summary.initial_cost > 1e-10) ? 
           100.0 * (summary.initial_cost - summary.final_cost) / summary.initial_cost : 0.0);
  printf("║ Termination:          %-35s  ║\n", 
         ceres::TerminationTypeToString(summary.termination_type));
  printf("╠═══════════════════════════════════════════════════════════════╣\n");
  printf("║ NORMALIZED COSTS (per residual)                               ║\n");
  int total_visual_factors = diag.num_reprojection_mono_factors + 
                            diag.num_reprojection_stereo_factors + 
                            diag.num_reprojection_one_frame_factors;
  int total_residuals = diag.total_marginalization_residuals + 
                       diag.total_imu_residuals + 
                       total_visual_factors * 2 + 
                       diag.total_depth_residuals;
  if (total_residuals > 0) {
    printf("║ Avg residual (initial): %.4e                            ║\n", 
           summary.initial_cost / total_residuals);
    printf("║ Avg residual (final):   %.4e                            ║\n", 
           summary.final_cost / total_residuals);
  }
  printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
  
  // Print per-group cost breakdown
  diag.printCostBreakdown();
  
  // Export to CSV for later visualization
  diag.exportToCSV("/datasets/optimization_dump.csv",
                   Headers[frame_count],  // timestamp of latest frame in window
                   inputImageCnt,         // frame id
                   summary.total_time_in_seconds * 1000.0,
                   (int)summary.iterations.size(),
                   ceres::TerminationTypeToString(summary.termination_type));

  ROS_INFO("Solver Time: %.2fms | Cost: %.2e -> %.2e | Iter: %d", 
           summary.total_time_in_seconds * 1000.0,
           summary.initial_cost,
           summary.final_cost,
           (int)summary.iterations.size());
  cout << summary.BriefReport() << endl;
  ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
  // printf("solver costs: %f \n", t_solver.toc());

  double2vector();
  //std::cout << "\n========== ESTIMATED SCALE & SHIFT ==========" << std::endl;
  //  std::cout << std::fixed << std::setprecision(5); // Set precision for cleaner output
  //  for (int i = 0; i <= WINDOW_SIZE; i++) {
  //      // para_ScaleShift[i][0] is Scale
  //      // para_ScaleShift[i][1] is Shift
  //      std::cout << "Frame [" << i << "]: "
  //                << "Scale = " << para_ScaleShift[i][0] << "  "
  //                << "Shift = " << para_ScaleShift[i][1] << std::endl;
  //  }
  //  std::cout << "===========================================\n" << std::endl;
  //// printf("frame_count: %d \n", frame_count);

  if (frame_count < WINDOW_SIZE) return;

  TicToc t_whole_marginalization;
  if (marginalization_flag == MARGIN_OLD) {
    MarginalizationInfo *marginalization_info = new MarginalizationInfo();
    vector2double();

    if (last_marginalization_info && last_marginalization_info->valid) {
      vector<int> drop_set;
      for (int i = 0;
           i < static_cast<int>(last_marginalization_parameter_blocks.size());
           i++) {
        if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
            last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
          drop_set.push_back(i);
      }
      // construct new marginlization_factor
      MarginalizationFactor *marginalization_factor =
          new MarginalizationFactor(last_marginalization_info);
      ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
          marginalization_factor, NULL, last_marginalization_parameter_blocks,
          drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }

    if (params.use_imu) {
      if (pre_integrations[1]->sum_dt < 10.0) {
        IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
            imu_factor, NULL,
            vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1],
                             para_SpeedBias[1]},
            vector<int>{0, 1});
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }

    {
      int feature_index = -1;
      ceres::LossFunction *depth_align_loss_function = new ceres::HuberLoss(0.1);
      // Just to use for debug later
      for (auto &it_per_id : f_manager.feature) {
      it_per_id.used_num = it_per_id.feature_per_frame.size();
      if (it_per_id.used_num < 4) continue;
      int first_frame_idx = it_per_id.start_frame;
      numbers[first_frame_idx]++;
      }

      for (auto &it_per_id : f_manager.feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4) continue;
        
        ++feature_index;
        
        int imu_i = it_per_id.start_frame;
        int imu_j = imu_i - 1;
        if (imu_i != 0) continue;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame) {
          imu_j++;
          if (imu_i != imu_j) {
            Vector3d pts_j = it_per_frame.point;
            auto *f_td = new ProjectionTwoFrameOneCamFactor(
                pts_i, pts_j, it_per_id.feature_per_frame[0].velocity,
                it_per_frame.velocity, it_per_id.feature_per_frame[0].cur_td,
                it_per_frame.cur_td);
            auto *residual_block_info = new ResidualBlockInfo(
                f_td, loss_function,
                vector<double *>{para_Pose[imu_i], para_Pose[imu_j],
                                 para_Ex_Pose[0], para_Feature[feature_index],
                                 para_Td[0]},
                vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual_block_info);
          }
          if (params.stereo && it_per_frame.is_stereo) {
            Vector3d pts_j_right = it_per_frame.pointRight;
            if (imu_i != imu_j) {
              auto *f = new ProjectionTwoFrameTwoCamFactor(
                  pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity,
                  it_per_frame.velocityRight,
                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
              auto *residual_block_info = new ResidualBlockInfo(
                  f, loss_function,
                  vector<double *>{para_Pose[imu_i], para_Pose[imu_j],
                                   para_Ex_Pose[0], para_Ex_Pose[1],
                                   para_Feature[feature_index], para_Td[0]},
                  vector<int>{0, 4});
              marginalization_info->addResidualBlockInfo(residual_block_info);
            } else {
              auto *f = new ProjectionOneFrameTwoCamFactor(
                  pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity,
                  it_per_frame.velocityRight,
                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
              auto *residual_block_info = new ResidualBlockInfo(
                  f, loss_function,
                  vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1],
                                   para_Feature[feature_index], para_Td[0]},
                  vector<int>{2});
              marginalization_info->addResidualBlockInfo(residual_block_info);
            }
          }
        }
      }
    }

    TicToc t_pre_margin;
    marginalization_info->preMarginalize();
    ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

    TicToc t_margin;
    marginalization_info->marginalize();
    ROS_DEBUG("marginalization %f ms", t_margin.toc());

    std::unordered_map<int64_t, double *> addr_shift;
    for (int i = 1; i <= WINDOW_SIZE; i++) {
      addr_shift[reinterpret_cast<int64_t>(para_Pose[i])] = para_Pose[i - 1];
      if (params.use_imu)
        addr_shift[reinterpret_cast<int64_t>(para_SpeedBias[i])] =
            para_SpeedBias[i - 1];
    }
    for (int i = 0; i < params.num_of_cam; i++)
      addr_shift[reinterpret_cast<int64_t>(para_Ex_Pose[i])] = para_Ex_Pose[i];

    addr_shift[reinterpret_cast<int64_t>(para_Td[0])] = para_Td[0];

    vector<double *> parameter_blocks =
        marginalization_info->getParameterBlocks(addr_shift);

    delete last_marginalization_info;
    last_marginalization_info = marginalization_info;
    last_marginalization_parameter_blocks = parameter_blocks;

  } else {
    if (last_marginalization_info &&
        std::count(std::begin(last_marginalization_parameter_blocks),
                   std::end(last_marginalization_parameter_blocks),
                   para_Pose[WINDOW_SIZE - 1])) {
      auto *marginalization_info = new MarginalizationInfo();
      vector2double();
      if (last_marginalization_info && last_marginalization_info->valid) {
        vector<int> drop_set;
        for (int i = 0;
             i < static_cast<int>(last_marginalization_parameter_blocks.size());
             i++) {
          ROS_ASSERT(last_marginalization_parameter_blocks[i] !=
                     para_SpeedBias[WINDOW_SIZE - 1]);
          if (last_marginalization_parameter_blocks[i] ==
              para_Pose[WINDOW_SIZE - 1])
            drop_set.push_back(i);
        }
        // construct new marginlization_factor
        auto *marginalization_factor =
            new MarginalizationFactor(last_marginalization_info);
        auto *residual_block_info = new ResidualBlockInfo(
            marginalization_factor, NULL, last_marginalization_parameter_blocks,
            drop_set);

        marginalization_info->addResidualBlockInfo(residual_block_info);
      }

      TicToc t_pre_margin;
      ROS_DEBUG("begin marginalization");
      marginalization_info->preMarginalize();
      ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

      TicToc t_margin;
      ROS_DEBUG("begin marginalization");
      marginalization_info->marginalize();
      ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

      std::unordered_map<int64_t, double *> addr_shift;
      for (int i = 0; i <= WINDOW_SIZE; i++) {
        if (i == WINDOW_SIZE - 1) continue;
        if (i == WINDOW_SIZE) {
          addr_shift[reinterpret_cast<int64_t>(para_Pose[i])] =
              para_Pose[i - 1];
          if (params.use_imu)
            addr_shift[reinterpret_cast<int64_t>(para_SpeedBias[i])] =
                para_SpeedBias[i - 1];
        } else {
          addr_shift[reinterpret_cast<int64_t>(para_Pose[i])] = para_Pose[i];
          if (params.use_imu)
            addr_shift[reinterpret_cast<int64_t>(para_SpeedBias[i])] =
                para_SpeedBias[i];
        }
      }
      for (int i = 0; i < params.num_of_cam; i++)
        addr_shift[reinterpret_cast<int64_t>(para_Ex_Pose[i])] =
            para_Ex_Pose[i];

      addr_shift[reinterpret_cast<int64_t>(para_Td[0])] = para_Td[0];

      vector<double *> parameter_blocks =
          marginalization_info->getParameterBlocks(addr_shift);
      delete last_marginalization_info;
      last_marginalization_info = marginalization_info;
      last_marginalization_parameter_blocks = parameter_blocks;
    }
  }
  // printf("whole marginalization costs: %f \n",
  // t_whole_marginalization.toc()); printf("whole time for ceres: %f \n",
  // t_whole.toc());
}

void Estimator::slideWindow() {
  TicToc t_margin;
  if (marginalization_flag == MARGIN_OLD) {
    double t_0 = Headers[0];
    back_R0 = Rs[0];
    back_P0 = Ps[0];
    if (frame_count == WINDOW_SIZE) {
      for (int i = 0; i < WINDOW_SIZE; i++) {
        Headers[i] = Headers[i + 1];
        Rs[i].swap(Rs[i + 1]);
        Ps[i].swap(Ps[i + 1]);
        if (params.use_imu) {
          std::swap(pre_integrations[i], pre_integrations[i + 1]);

          dt_buf[i].swap(dt_buf[i + 1]);
          linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
          angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

          Vs[i].swap(Vs[i + 1]);
          Bas[i].swap(Bas[i + 1]);
          Bgs[i].swap(Bgs[i + 1]);
        }
      }
      Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
      Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
      Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

      if (params.use_imu) {
        Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
        Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
        Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

        delete pre_integrations[WINDOW_SIZE];
        pre_integrations[WINDOW_SIZE] = new IntegrationBase{
            acc_0,        gyr_0,        Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE],
            params.acc_n, params.gyr_n, params.acc_w,     params.gyr_w,
            params.g};

        dt_buf[WINDOW_SIZE].clear();
        linear_acceleration_buf[WINDOW_SIZE].clear();
        angular_velocity_buf[WINDOW_SIZE].clear();
      }

      {
        map<double, ImageFrame>::iterator it_0;
        it_0 = all_image_frame.find(t_0);
        delete it_0->second.pre_integration;
        all_image_frame.erase(all_image_frame.begin(), it_0);
      }
      slideWindowOld();
    }
  } else {
    if (frame_count == WINDOW_SIZE) {
      Headers[frame_count - 1] = Headers[frame_count];
      Ps[frame_count - 1] = Ps[frame_count];
      Rs[frame_count - 1] = Rs[frame_count];

      if (params.use_imu) {
        for (unsigned i = 0; i < dt_buf[frame_count].size(); i++) {
          double tmp_dt = dt_buf[frame_count][i];
          Vector3d tmp_linear_acceleration =
              linear_acceleration_buf[frame_count][i];
          Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

          pre_integrations[frame_count - 1]->push_back(
              tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

          dt_buf[frame_count - 1].push_back(tmp_dt);
          linear_acceleration_buf[frame_count - 1].push_back(
              tmp_linear_acceleration);
          angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
        }

        Vs[frame_count - 1] = Vs[frame_count];
        Bas[frame_count - 1] = Bas[frame_count];
        Bgs[frame_count - 1] = Bgs[frame_count];

        delete pre_integrations[WINDOW_SIZE];
        pre_integrations[WINDOW_SIZE] = new IntegrationBase{
            acc_0,        gyr_0,        Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE],
            params.acc_n, params.gyr_n, params.acc_w,     params.gyr_w,
            params.g};

        dt_buf[WINDOW_SIZE].clear();
        linear_acceleration_buf[WINDOW_SIZE].clear();
        angular_velocity_buf[WINDOW_SIZE].clear();
      }
      slideWindowNew();
    }
  }
}

void Estimator::slideWindowNew() {
  sum_of_front++;
  f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld() {
  sum_of_back++;

  bool shift_depth = solver_flag == NON_LINEAR;
  if (shift_depth) {
    Matrix3d R0;
    Matrix3d R1;
    Vector3d P0;
    Vector3d P1;
    R0 = back_R0 * ric[0];
    R1 = Rs[0] * ric[0];
    P0 = back_P0 + back_R0 * tic[0];
    P1 = Ps[0] + Rs[0] * tic[0];
    f_manager.removeBackShiftDepth(R0, P0, R1, P1);
  } else
    f_manager.removeBack();
}

void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T) {
  T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = Rs[frame_count];
  T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T) {
  T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = Rs[index];
  T.block<3, 1>(0, 3) = Ps[index];
}

map<int, Eigen::Vector3d> Estimator::predictPtsInNextFrame() {
  // printf("predict pts in next frame\n");
  if (frame_count < 2) return map<int, Eigen::Vector3d>();
  // predict next pose. Assume constant velocity motion
  Eigen::Matrix4d curT;
  Eigen::Matrix4d prevT;
  Eigen::Matrix4d nextT;
  getPoseInWorldFrame(curT);
  getPoseInWorldFrame(frame_count - 1, prevT);
  nextT = curT * (prevT.inverse() * curT);
  map<int, Eigen::Vector3d> predictPts;

  for (auto &it_per_id : f_manager.feature) {
    if (it_per_id.estimated_depth > 0) {
      int firstIndex = it_per_id.start_frame;
      int lastIndex =
          it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
      // printf("cur frame index  %d last frame index %d\n", frame_count,
      // lastIndex);
      if (static_cast<int>(it_per_id.feature_per_frame.size()) >= 2 &&
          lastIndex == frame_count) {
        double depth = it_per_id.estimated_depth;
        Vector3d pts_j =
            ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
        Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
        Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() *
                             (pts_w - nextT.block<3, 1>(0, 3));
        Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
        int ptsIndex = it_per_id.feature_id;
        predictPts[ptsIndex] = pts_cam;
      }
    }
  }
  // featureTracker.setPrediction(predictPts);
  // printf("estimator output %d predict pts\n",(int)predictPts.size());
  return predictPts;
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici,
                                    Vector3d &tici, Matrix3d &Rj, Vector3d &Pj,
                                    Matrix3d &ricj, Vector3d &ticj,
                                    double depth, Vector3d &uvi,
                                    Vector3d &uvj) {
  Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
  Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
  Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
  double rx = residual.x();
  double ry = residual.y();
  return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex) {
  // return;
  int feature_index = -1;
  for (auto &it_per_id : f_manager.feature) {
    double err = 0;
    int errCnt = 0;
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4) continue;
    feature_index++;
    int imu_i = it_per_id.start_frame;
    int imu_j = imu_i - 1;
    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
    double depth = it_per_id.estimated_depth;
    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      if (imu_i != imu_j) {
        Vector3d pts_j = it_per_frame.point;
        double tmp_error =
            reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j],
                              Ps[imu_j], ric[0], tic[0], depth, pts_i, pts_j);
        err += tmp_error;
        errCnt++;
        // printf("tmp_error %f\n", params.focal_length / 1.5 * tmp_error);
      }
      // need to rewrite projecton factor.........
      if (params.stereo && it_per_frame.is_stereo) {
        Vector3d pts_j_right = it_per_frame.pointRight;
        if (imu_i != imu_j) {
          double tmp_error = reprojectionError(
              Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j],
              ric[1], tic[1], depth, pts_i, pts_j_right);
          err += tmp_error;
          errCnt++;
          // printf("tmp_error %f\n", params.focal_length / 1.5 * tmp_error);
        } else {
          double tmp_error = reprojectionError(
              Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j],
              ric[1], tic[1], depth, pts_i, pts_j_right);
          err += tmp_error;
          errCnt++;
          // printf("tmp_error %f\n", params.focal_length / 1.5 * tmp_error);
        }
      }
    }
    double ave_err = err / errCnt;
    if (ave_err * params.focal_length > 3)
      removeIndex.insert(it_per_id.feature_id);
  }
}

void Estimator::fastPredictIMU(double t,
                               const Eigen::Vector3d &linear_acceleration,
                               const Eigen::Vector3d &angular_velocity) {
  double dt = t - latest_time;
  latest_time = t;
  Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
  Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
  latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
  Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
  latest_V = latest_V + dt * un_acc;
  latest_acc_0 = linear_acceleration;
  latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates() {
  mPropagate.lock();
  latest_time = Headers[frame_count] + td;
  latest_P = Ps[frame_count];
  latest_Q = Rs[frame_count];
  latest_V = Vs[frame_count];
  latest_Ba = Bas[frame_count];
  latest_Bg = Bgs[frame_count];
  latest_acc_0 = acc_0;
  latest_gyr_0 = gyr_0;
  mBuf.lock();
  queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
  queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
  mBuf.unlock();
  while (!tmp_accBuf.empty()) {
    double t = tmp_accBuf.front().first;
    Eigen::Vector3d acc = tmp_accBuf.front().second;
    Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
    fastPredictIMU(t, acc, gyr);
    tmp_accBuf.pop();
    tmp_gyrBuf.pop();
  }
  mPropagate.unlock();
}

}  // namespace vins::estimator
