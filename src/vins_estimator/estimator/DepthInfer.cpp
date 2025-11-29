#include "vins_estimator/estimator/DepthInfer.h"
#include <fstream>
#include <iostream>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only log Warnings and Errors to keep console clean
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

DepthInfer::DepthInfer(std::string engine_path) {
    std::cout << "Loading TensorRT Engine: " << engine_path << std::endl;

    // 1. Load Engine File
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) { 
        std::cerr << "Error loading engine file! Check path." << std::endl; 
        return; 
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    // 2. Create Runtime
    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) std::cerr << "Failed to create TRT Runtime!" << std::endl;

    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    if (!engine) std::cerr << "Failed to deserialize Engine!" << std::endl;

    context = engine->createExecutionContext();
    delete[] trtModelStream;

    // 3. Allocate GPU Memory
    cudaMalloc(&buffers[0], INPUT_SIZE * sizeof(float)); // Input
    cudaMalloc(&buffers[1], OUTPUT_SIZE * sizeof(float)); // Output
    
    cpu_output_buffer = new float[OUTPUT_SIZE];
}

// Destructor
DepthInfer::~DepthInfer() {
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete[] cpu_output_buffer;
    
    // Clean up TRT pointers
    // Note: In newer TRT versions, use delete. In older, use ->destroy()
    if(context) delete context;
    if(engine) delete engine;
    if(runtime) delete runtime;
}

cv::Mat DepthInfer::infer(cv::Mat& img) {
    if (!context) {
        std::cerr << "Inference skipped: Context not initialized." << std::endl;
        return cv::Mat();
    }

    // --- PREPROCESS ---
    cv::Mat resized, float_img;
    cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
    
    // Convert to float (0..1)
    resized.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);
    
    // Normalize (ImageNet Mean/Std)
    // Mean: 0.485, 0.456, 0.406 | Std: 0.229, 0.224, 0.225
    cv::subtract(float_img, cv::Scalar(0.485, 0.456, 0.406), float_img);
    cv::divide(float_img, cv::Scalar(0.229, 0.224, 0.225), float_img);

    // HWC to CHW conversion
    std::vector<float> input_nchw(INPUT_SIZE);
    int idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < INPUT_H; ++h) {
            for (int w = 0; w < INPUT_W; ++w) {
                // OpenCV is BGR. Model needs RGB.
                // input[c] corresponds to RGB. BGR image index 2-c flips it.
                input_nchw[idx++] = float_img.at<cv::Vec3f>(h, w)[2 - c]; 
            }
        }
    }

    // --- INFERENCE ---
    cudaMemcpy(buffers[0], input_nchw.data(), INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // executeV2 is standard for recent TensorRT versions
    context->executeV2(buffers); 

    cudaMemcpy(cpu_output_buffer, buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // --- POSTPROCESS ---
    // Wrap buffer in Mat
    cv::Mat depth_map(INPUT_H, INPUT_W, CV_32FC1, cpu_output_buffer);
    
    // Return a deep copy so we can reuse the buffer next time
    return depth_map.clone(); 
}
