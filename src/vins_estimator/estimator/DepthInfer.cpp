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
    std::cout << "\n========== TRT ENGINE TENSORS (New API) ==========" << std::endl;
    int nbIOTensors = engine->getNbIOTensors();
    
    for (int i = 0; i < nbIOTensors; i++) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::Dims dims = engine->getTensorShape(name);
        nvinfer1::DataType dtype = engine->getTensorDataType(name);
        nvinfer1::TensorIOMode ioMode = engine->getTensorIOMode(name);
        bool isInput = (ioMode == nvinfer1::TensorIOMode::kINPUT);
        
        std::cout << "Index " << i << " Name: [" << name << "] " 
                  << (isInput ? "INPUT" : "OUTPUT") << " | ";
        
        std::cout << "Type: " << (dtype == nvinfer1::DataType::kFLOAT ? "FP32" : 
                                  dtype == nvinfer1::DataType::kHALF ? "FP16" : "Other") << " | ";

        std::cout << "Dims: [";
        for (int d = 0; d < dims.nbDims; d++) {
            std::cout << dims.d[d] << (d < dims.nbDims - 1 ? "x" : "");
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "==================================================\n" << std::endl;
    context = engine->createExecutionContext();
    delete[] trtModelStream;

    // 3. Allocate GPU Memory
    cudaMalloc(&buffers[0], INPUT_SIZE * sizeof(float)); // Input
    cudaMalloc(&buffers[1], OUTPUT_SIZE * sizeof(float)); // Output
    cudaMallocHost((void**)&h_input_pinned, INPUT_SIZE * sizeof(float));
    cudaStreamCreate(&stream);
    cpu_output_buffer = new float[OUTPUT_SIZE];
}

// Destructor
DepthInfer::~DepthInfer() {
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete[] cpu_output_buffer;
    if (h_input_pinned) cudaFreeHost(h_input_pinned);
    // Clean up TRT pointers
    // Note: In newer TRT versions, use delete. In older, use ->destroy()
    if(context) delete context;
    if(engine) delete engine;
    if(runtime) delete runtime;
    
}

cv::Mat DepthInfer::infer(cv::Mat& img) {
    if (!context) {
        std::cerr << "Context not initialized." << std::endl;
        return cv::Mat();
    }

    // 1. Resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));

    // 2. Optimized Preprocessing (Single Loop, Pinned Memory)
    // Constants for Normalization
    const float mean_r = 0.485f; const float std_r = 0.229f;
    const float s_r = 1.0f / (255.0f * std_r);
    const float o_r = mean_r / std_r;

    const float mean_g = 0.456f; const float std_g = 0.224f;
    const float s_g = 1.0f / (255.0f * std_g);
    const float o_g = mean_g / std_g;

    const float mean_b = 0.406f; const float std_b = 0.225f;
    const float s_b = 1.0f / (255.0f * std_b);
    const float o_b = mean_b / std_b;

    // Pointers to pinned memory planes
    float* p_r = h_input_pinned;
    float* p_g = h_input_pinned + (INPUT_W * INPUT_H);
    float* p_b = h_input_pinned + (2 * INPUT_W * INPUT_H);

    int total_pixels = INPUT_W * INPUT_H;
    const uchar* ptr_img = resized.ptr<uchar>(0);

    // Fast HWC -> CHW + Normalize loop
    for (int i = 0; i < total_pixels; ++i) {
        uchar b = ptr_img[3*i + 0];
        uchar g = ptr_img[3*i + 1];
        uchar r = ptr_img[3*i + 2];

        p_r[i] = (static_cast<float>(r) * s_r) - o_r;
        p_g[i] = (static_cast<float>(g) * s_g) - o_g;
        p_b[i] = (static_cast<float>(b) * s_b) - o_b;
    }

    // --- INFERENCE ---
    
    // 1. Upload asynchronously
    cudaMemcpyAsync(buffers[0], h_input_pinned, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // 2. Wait for upload to finish (Safe for executeV2)
    cudaStreamSynchronize(stream);

    // 3. Execute (Synchronous/Blocking is safer if enqueueV2 is missing)
    context->executeV2(buffers);

    // 4. Download asynchronously
    cudaMemcpyAsync(cpu_output_buffer, buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    // 5. Wait for download
    cudaStreamSynchronize(stream);

    // --- POSTPROCESS ---
    cv::Mat depth_map(OUTPUT_H, OUTPUT_W, CV_32FC1, cpu_output_buffer);
    return depth_map.clone(); 
}
