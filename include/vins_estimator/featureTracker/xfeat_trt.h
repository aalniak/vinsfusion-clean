#pragma once
#include <NvInfer.h>

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace vins::estimator {

// Result of one XFeat sparse extraction. Keypoints are returned in the ORIGINAL
// image coordinate frame (the engine runs at a fixed ÷32 resolution; we rescale).
struct XFeatFeatures {
  std::vector<cv::Point2f> keypoints;  // size N
  std::vector<float> scores;           // size N
  std::vector<float> descriptors;      // size N*64, row-major, L2-normalized
  int n = 0;                           // == engine top_k
};

// Thin TensorRT wrapper around the static XFeat extractor engine exported by
// export/export_xfeat.py. Single image in -> {keypoints, scores, descriptors}.
class XFeatTRT {
 public:
  explicit XFeatTRT(const std::string &engine_path);
  ~XFeatTRT();

  XFeatTRT(const XFeatTRT &) = delete;
  XFeatTRT &operator=(const XFeatTRT &) = delete;

  // img: 8-bit grayscale or BGR, any size (resized internally to the engine size).
  XFeatFeatures run(const cv::Mat &img);

  int engineWidth() const { return in_w_; }
  int engineHeight() const { return in_h_; }
  int topK() const { return top_k_; }

 private:
  void loadEngine(const std::string &path);
  void allocateBuffers();

  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  void *stream_ = nullptr;

  // Engine I/O geometry (read from the engine at load time).
  int in_w_ = 0, in_h_ = 0;  // network input W,H (÷32)
  int top_k_ = 0;            // fixed number of keypoints

  // Device buffers.
  void *d_image_ = nullptr;
  void *d_kpts_ = nullptr;
  void *d_scores_ = nullptr;
  void *d_desc_ = nullptr;

  // Tensor names must match the ONNX export.
  static constexpr const char *kIn = "image";
  static constexpr const char *kKpts = "keypoints";
  static constexpr const char *kScores = "scores";
  static constexpr const char *kDesc = "descriptors";
};

}  // namespace vins::estimator
