#include "vins_estimator/featureTracker/lighterglue_trt.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <fstream>
#include <iostream>

#include "vins_estimator/estimator/TrtCommon.h"

#define CHECK_CUDA(status)                                                     \
  do {                                                                         \
    auto _s = (status);                                                        \
    if (_s != 0) {                                                             \
      std::cerr << "[LighterGlueTRT] CUDA failure " << _s << " at " << __LINE__ \
                << std::endl;                                                  \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

namespace vins::estimator {

LighterGlueTRT::LighterGlueTRT(const std::string &engine_path) {
  cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&stream_));
  loadEngine(engine_path);
  allocateBuffers();
}

LighterGlueTRT::~LighterGlueTRT() {
  if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
  if (d_kpts0_) cudaFree(d_kpts0_);
  if (d_desc0_) cudaFree(d_desc0_);
  if (d_kpts1_) cudaFree(d_kpts1_);
  if (d_desc1_) cudaFree(d_desc1_);
  if (d_matches0_) cudaFree(d_matches0_);
  if (d_mscores0_) cudaFree(d_mscores0_);
}

void LighterGlueTRT::loadEngine(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.good()) {
    std::cerr << "[LighterGlueTRT] cannot open engine: " << path << std::endl;
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
    std::cerr << "[LighterGlueTRT] failed to deserialize engine" << std::endl;
    std::abort();
  }
  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext(),
      [](nvinfer1::IExecutionContext *c) { delete c; });

  auto k0 = engine_->getTensorShape(kKpts0);  // [1, N, 2]
  n_ = k0.d[1];
  std::cerr << "[LighterGlueTRT] engine num_kpts=" << n_ << std::endl;
}

void LighterGlueTRT::allocateBuffers() {
  CHECK_CUDA(cudaMalloc(&d_kpts0_, static_cast<size_t>(n_) * 2 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_desc0_, static_cast<size_t>(n_) * 64 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_kpts1_, static_cast<size_t>(n_) * 2 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_desc1_, static_cast<size_t>(n_) * 64 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_matches0_, static_cast<size_t>(n_) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_mscores0_, static_cast<size_t>(n_) * sizeof(float)));
}

LGMatches LighterGlueTRT::run(const std::vector<cv::Point2f> &kpts0,
                              const std::vector<float> &desc0,
                              const std::vector<cv::Point2f> &kpts1,
                              const std::vector<float> &desc1) {
  if (static_cast<int>(kpts0.size()) != n_ ||
      static_cast<int>(kpts1.size()) != n_ ||
      static_cast<int>(desc0.size()) != n_ * 64 ||
      static_cast<int>(desc1.size()) != n_ * 64) {
    std::cerr << "[LighterGlueTRT] input size mismatch: expected N=" << n_
              << " got kpts0=" << kpts0.size() << " desc0=" << desc0.size()
              << std::endl;
    std::abort();
  }

  auto stream = static_cast<cudaStream_t>(stream_);
  // cv::Point2f is two contiguous floats, so kpts.data() is a valid [N,2] buffer.
  CHECK_CUDA(cudaMemcpyAsync(d_kpts0_, kpts0.data(), n_ * 2 * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_desc0_, desc0.data(), n_ * 64 * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_kpts1_, kpts1.data(), n_ * 2 * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_desc1_, desc1.data(), n_ * 64 * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  context_->setTensorAddress(kKpts0, d_kpts0_);
  context_->setTensorAddress(kDesc0, d_desc0_);
  context_->setTensorAddress(kKpts1, d_kpts1_);
  context_->setTensorAddress(kDesc1, d_desc1_);
  context_->setTensorAddress(kMatches0, d_matches0_);
  context_->setTensorAddress(kMscores0, d_mscores0_);
  context_->enqueueV3(stream);

  LGMatches out;
  out.n = n_;
  out.matches0.resize(n_);
  out.mscores0.resize(n_);
  CHECK_CUDA(cudaMemcpyAsync(out.matches0.data(), d_matches0_, n_ * sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaMemcpyAsync(out.mscores0.data(), d_mscores0_, n_ * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return out;
}

}  // namespace vins::estimator
