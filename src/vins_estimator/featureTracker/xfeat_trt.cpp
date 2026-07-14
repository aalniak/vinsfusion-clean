#include "vins_estimator/featureTracker/xfeat_trt.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>

#include "vins_estimator/estimator/TrtCommon.h"

#define CHECK_CUDA(status)                                            \
  do {                                                               \
    auto _s = (status);                                              \
    if (_s != 0) {                                                   \
      std::cerr << "[XFeatTRT] CUDA failure " << _s << " at " << __LINE__ \
                << std::endl;                                        \
      std::abort();                                                  \
    }                                                                \
  } while (0)

namespace vins::estimator {

XFeatTRT::XFeatTRT(const std::string &engine_path) {
  cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&stream_));
  loadEngine(engine_path);
  allocateBuffers();
}

XFeatTRT::~XFeatTRT() {
  if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
  if (d_image_) cudaFree(d_image_);
  if (d_kpts_) cudaFree(d_kpts_);
  if (d_scores_) cudaFree(d_scores_);
  if (d_desc_) cudaFree(d_desc_);
  if (d_dense_) cudaFree(d_dense_);
  if (dense_) cudaFreeHost(dense_);
}

void XFeatTRT::loadEngine(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.good()) {
    std::cerr << "[XFeatTRT] cannot open engine: " << path << std::endl;
    std::abort();
  }
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> blob(size);
  file.read(blob.data(), size);

  nvinfer1::IRuntime *runtime = TrtManager::getRuntime();
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(blob.data(), size),
      [](nvinfer1::ICudaEngine *e) { delete e; });
  if (!engine_) {
    std::cerr << "[XFeatTRT] failed to deserialize engine" << std::endl;
    std::abort();
  }
  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext(),
      [](nvinfer1::IExecutionContext *c) { delete c; });

  // Read fixed geometry from the engine (input is [1,1,H,W], kpts [1,K,2]).
  auto in_shape = engine_->getTensorShape(kIn);
  in_h_ = in_shape.d[2];
  in_w_ = in_shape.d[3];
  auto kpts_shape = engine_->getTensorShape(kKpts);
  top_k_ = kpts_shape.d[1];

  // Detect the optional dense descriptor map output (engine exported with --dense).
  int nio = engine_->getNbIOTensors();
  for (int i = 0; i < nio; ++i) {
    if (std::string(engine_->getIOTensorName(i)) == kDense) {
      has_dense_ = true;
      break;
    }
  }
  if (has_dense_) {
    auto ds = engine_->getTensorShape(kDense);  // (1,64,H/8,W/8)
    d8_h_ = ds.d[2];
    d8_w_ = ds.d[3];
  }
  std::cerr << "[XFeatTRT] engine " << in_w_ << "x" << in_h_ << " top_k=" << top_k_
            << (has_dense_ ? " +dense(" + std::to_string(d8_w_) + "x" +
                                 std::to_string(d8_h_) + ")"
                           : "")
            << std::endl;
}

void XFeatTRT::allocateBuffers() {
  CHECK_CUDA(cudaMalloc(&d_image_, static_cast<size_t>(in_w_) * in_h_ * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_kpts_, static_cast<size_t>(top_k_) * 2 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_scores_, static_cast<size_t>(top_k_) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_desc_, static_cast<size_t>(top_k_) * 64 * sizeof(float)));
  if (has_dense_) {
    dense_n_ = static_cast<size_t>(64) * d8_h_ * d8_w_;
    CHECK_CUDA(cudaMalloc(&d_dense_, dense_n_ * sizeof(float)));
    // Pinned host memory -> fast, truly-async D2H (pageable 4 MB copies stalled the
    // front-end and broke VINS init).
    CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&dense_),
                             dense_n_ * sizeof(float), cudaHostAllocDefault));
  }
}

XFeatFeatures XFeatTRT::run(const cv::Mat &img) {
  // 1. Preprocess: grayscale -> engine size -> float32. XFeat's InstanceNorm makes
  //    the input scale/shift invariant, so we keep the raw 0..255 range (matches the
  //    export-time validation).
  cv::Mat gray;
  if (img.channels() == 3) {
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = img;
  }
  cv::Mat resized;
  if (gray.cols != in_w_ || gray.rows != in_h_) {
    cv::resize(gray, resized, cv::Size(in_w_, in_h_));
  } else {
    resized = gray;
  }
  cv::Mat f32;
  resized.convertTo(f32, CV_32FC1);

  auto stream = static_cast<cudaStream_t>(stream_);
  CHECK_CUDA(cudaMemcpyAsync(d_image_, f32.ptr<float>(),
                             static_cast<size_t>(in_w_) * in_h_ * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  context_->setTensorAddress(kIn, d_image_);
  context_->setTensorAddress(kKpts, d_kpts_);
  context_->setTensorAddress(kScores, d_scores_);
  context_->setTensorAddress(kDesc, d_desc_);
  if (has_dense_) context_->setTensorAddress(kDense, d_dense_);
  context_->enqueueV3(stream);

  XFeatFeatures out;
  out.n = top_k_;
  std::vector<float> kpts(static_cast<size_t>(top_k_) * 2);
  out.scores.resize(top_k_);
  out.descriptors.resize(static_cast<size_t>(top_k_) * 64);

  CHECK_CUDA(cudaMemcpyAsync(kpts.data(), d_kpts_, kpts.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaMemcpyAsync(out.scores.data(), d_scores_,
                             out.scores.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaMemcpyAsync(out.descriptors.data(), d_desc_,
                             out.descriptors.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  if (has_dense_ && dense_enabled_) {
    CHECK_CUDA(cudaMemcpyAsync(dense_, d_dense_, dense_n_ * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    // image-coord -> dense-cell: cell = x * (W/8)/img_w  (engine resize folded in).
    dense_sx_ = static_cast<float>(d8_w_) / static_cast<float>(img.cols);
    dense_sy_ = static_cast<float>(d8_h_) / static_cast<float>(img.rows);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // 2. Rescale keypoints from engine resolution back to the original image.
  const float sx = static_cast<float>(img.cols) / static_cast<float>(in_w_);
  const float sy = static_cast<float>(img.rows) / static_cast<float>(in_h_);
  out.keypoints.reserve(top_k_);
  for (int i = 0; i < top_k_; ++i) {
    out.keypoints.emplace_back(kpts[2 * i] * sx, kpts[2 * i + 1] * sy);
  }
  return out;
}

std::array<float, 64> XFeatTRT::sampleDense(float x, float y) const {
  std::array<float, 64> out{};
  if (!has_dense_ || dense_ == nullptr) return out;
  float cx = x * dense_sx_, cy = y * dense_sy_;  // continuous cell coords
  if (cx < 0.f) cx = 0.f;
  if (cy < 0.f) cy = 0.f;
  if (cx > d8_w_ - 1) cx = d8_w_ - 1;
  if (cy > d8_h_ - 1) cy = d8_h_ - 1;
  int x0 = static_cast<int>(cx), y0 = static_cast<int>(cy);
  int x1 = std::min(x0 + 1, d8_w_ - 1), y1 = std::min(y0 + 1, d8_h_ - 1);
  float ax = cx - x0, ay = cy - y0;
  float w00 = (1 - ax) * (1 - ay), w01 = ax * (1 - ay);
  float w10 = (1 - ax) * ay, w11 = ax * ay;
  const float *D = dense_;
  const int s = d8_h_ * d8_w_;
  const int i00 = y0 * d8_w_ + x0, i01 = y0 * d8_w_ + x1;
  const int i10 = y1 * d8_w_ + x0, i11 = y1 * d8_w_ + x1;
  float norm = 0.f;
  for (int c = 0; c < 64; ++c) {
    const float *Dc = D + c * s;
    float v = w00 * Dc[i00] + w01 * Dc[i01] + w10 * Dc[i10] + w11 * Dc[i11];
    out[c] = v;
    norm += v * v;
  }
  norm = std::sqrt(norm) + 1e-9f;
  for (int c = 0; c < 64; ++c) out[c] /= norm;
  return out;
}

}  // namespace vins::estimator
