#include "vins_estimator/estimator/SplgInference.h"
#include "vins_estimator/estimator/TrtCommon.h"
#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>

// CUDA Error Check Helper
#define CHECK_CUDA(status) \
    if (status != 0) { std::cerr << "Cuda failure: " << status << std::endl; abort(); }



SplgInference::SplgInference(const std::string& engine_path) {
    cudaStreamCreate((cudaStream_t*)&stream_);
    if (!loadEngine(engine_path)) {
        std::cerr << "Failed to load SP+LG Engine!" << std::endl;
    }
    prepareBuffers();
}

SplgInference::~SplgInference() {
    cudaStreamDestroy((cudaStream_t)stream_);
    if (d_img0_) cudaFree(d_img0_);
    if (d_img1_) cudaFree(d_img1_);
    if (d_kpts0_) cudaFree(d_kpts0_);
    if (d_kpts1_) cudaFree(d_kpts1_);
    if (d_matches_) cudaFree(d_matches_);
    if (d_scores_) cudaFree(d_scores_);
}

bool SplgInference::loadEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) return false;

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> trtModelStream(size);
    file.read(trtModelStream.data(), size);

    nvinfer1::IRuntime* runtime = TrtManager::getRuntime();
    
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
    runtime->deserializeCudaEngine(trtModelStream.data(), size), 
    [](nvinfer1::ICudaEngine* e) { delete e; }
);

    if (!engine_) return false;

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext(), [](nvinfer1::IExecutionContext* c) { delete c; });

    return true;
}

void SplgInference::prepareBuffers() {
    // 1. Calculate Sizes
    size_t img_size = input_w_ * input_h_ * sizeof(float);
    int max_kps = 512; // Ensure this matches your engine export settings

    // 2. Allocate Inputs
    CHECK_CUDA(cudaMalloc(&d_img0_, img_size));
    CHECK_CUDA(cudaMalloc(&d_img1_, img_size));

    // 3. Allocate Outputs
    CHECK_CUDA(cudaMalloc(&d_kpts0_,   max_kps * 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kpts1_,   max_kps * 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_matches_, max_kps * 2 * sizeof(int))); 
    CHECK_CUDA(cudaMalloc(&d_scores_,  max_kps * sizeof(float)));
}

MatchResult SplgInference::run(const cv::Mat& img1_raw, const cv::Mat& img2_raw) {
    // 1. Preprocessing
    cv::Mat img1, img2;
    
    // Resize to engine input size
    cv::resize(img1_raw, img1, cv::Size(input_w_, input_h_));
    cv::resize(img2_raw, img2, cv::Size(input_w_, input_h_));

    // Convert to Grayscale + Float + Normalize [0,1]
    if (img1.channels() == 3) cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
    if (img2.channels() == 3) cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);

    img1.convertTo(img1, CV_32FC1, 1.0 / 255.0);
    img2.convertTo(img2, CV_32FC1, 1.0 / 255.0);

    // 2. Copy to GPU (Async)
    size_t img_bytes = input_w_ * input_h_ * sizeof(float);
    cudaMemcpyAsync(d_img0_, img1.ptr<float>(), img_bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream_);
    cudaMemcpyAsync(d_img1_, img2.ptr<float>(), img_bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream_);

    // 2. Bind Pointers for TensorRT 10
    context_->setTensorAddress("image0", d_img0_);
    context_->setTensorAddress("image1", d_img1_);
    context_->setTensorAddress("kpts0",    d_kpts0_);
    context_->setTensorAddress("kpts1",    d_kpts1_);
    context_->setTensorAddress("matches0", d_matches_);
    context_->setTensorAddress("mscores0", d_scores_);

    // 3. Execute
    context_->enqueueV3((cudaStream_t)stream_);

    // 4. Copy Back Results (Use the new pointers)
    int max_kps = 2048;
    std::vector<int> matches0(max_kps * 2);
    std::vector<float> kpts0(max_kps * 2);
    std::vector<float> kpts1(max_kps * 2);
    std::vector<float> scores(max_kps);

    cudaMemcpyAsync(matches0.data(), d_matches_, matches0.size() * sizeof(int), cudaMemcpyDeviceToHost, (cudaStream_t)stream_);
    cudaMemcpyAsync(kpts0.data(),    d_kpts0_,   kpts0.size() * sizeof(float), cudaMemcpyDeviceToHost, (cudaStream_t)stream_);
    cudaMemcpyAsync(kpts1.data(),    d_kpts1_,   kpts1.size() * sizeof(float), cudaMemcpyDeviceToHost, (cudaStream_t)stream_);
    cudaMemcpyAsync(scores.data(),   d_scores_,  scores.size() * sizeof(float), cudaMemcpyDeviceToHost, (cudaStream_t)stream_);
    
    cudaStreamSynchronize((cudaStream_t)stream_);

    // 5. Parse Matches
    MatchResult result;
    
    // Calculate scale factors to map back to original resolution
    float sx = (float)img1_raw.cols / input_w_;
    float sy = (float)img1_raw.rows / input_h_;

    // End-to-end LightGlue usually outputs 'matches0' as indices into keypoints
    // Format varies by export script. 
    // Common: matches0 is (N, 2) where col 0 is index in kpts0, col 1 is index in kpts1
    
    for(int i = 0; i < max_kps; ++i) {
        // If matches are output directly as pairs:
        int idx0 = matches0[2*i];
        int idx1 = matches0[2*i + 1];

        if (idx0 == -1 || idx1 == -1) continue; // No match

        // Get coordinates
        float x1 = kpts0[2*idx0];
        float y1 = kpts0[2*idx0+1];
        float x2 = kpts1[2*idx1];
        float y2 = kpts1[2*idx1+1];

        // Scale back
        result.kps1.push_back(cv::Point2f(x1 * sx, y1 * sy));
        result.kps2.push_back(cv::Point2f(x2 * sx, y2 * sy));
        result.scores.push_back(scores[i]);
    }

    return result;
}