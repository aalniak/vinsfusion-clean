#pragma once
#include <NvInfer.h>

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace vins::estimator {

// Per-keypoint match result for image0 (the "from" image). matches0[i] is the index
// of the matched keypoint in image1, or -1 if unmatched; mscores0[i] in (0,1].
struct LGMatches {
  std::vector<int> matches0;    // size N
  std::vector<float> mscores0;  // size N
  int n = 0;
};

// TensorRT wrapper around the clean LighterGlue matcher engine exported by
// export/export_lighterglue.py. Inputs are XFeat keypoints (raw pixel coords at the
// export resolution) + 64-D descriptors for two images; output is matches0/mscores0.
class LighterGlueTRT {
 public:
  explicit LighterGlueTRT(const std::string &engine_path);
  ~LighterGlueTRT();

  LighterGlueTRT(const LighterGlueTRT &) = delete;
  LighterGlueTRT &operator=(const LighterGlueTRT &) = delete;

  // kpts*/desc* must each hold exactly numKpts() entries (desc is N*64).
  LGMatches run(const std::vector<cv::Point2f> &kpts0,
                const std::vector<float> &desc0,
                const std::vector<cv::Point2f> &kpts1,
                const std::vector<float> &desc1);

  int numKpts() const { return n_; }

 private:
  void loadEngine(const std::string &path);
  void allocateBuffers();

  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  void *stream_ = nullptr;
  int n_ = 0;  // fixed number of keypoints per image

  void *d_kpts0_ = nullptr;
  void *d_desc0_ = nullptr;
  void *d_kpts1_ = nullptr;
  void *d_desc1_ = nullptr;
  void *d_matches0_ = nullptr;
  void *d_mscores0_ = nullptr;

  static constexpr const char *kKpts0 = "kpts0";
  static constexpr const char *kDesc0 = "desc0";
  static constexpr const char *kKpts1 = "kpts1";
  static constexpr const char *kDesc1 = "desc1";
  static constexpr const char *kMatches0 = "matches0";
  static constexpr const char *kMscores0 = "mscores0";
};

}  // namespace vins::estimator
