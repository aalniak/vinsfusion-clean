#pragma once
#include <NvInfer.h>
#include <iostream>
#include <memory>
#include <mutex>

// A global logger (required by TensorRT)
class TrtLogger : public nvinfer1::ILogger {
public:
    // 'noexcept' is required for TensorRT 8+ and 10
    void log(Severity severity, const char* msg) noexcept override {
        // Only log warnings and errors to keep console clean
        if (severity <= Severity::kWARNING) 
            std::cout << "[TRT-Global] " << msg << std::endl;
    }
};

// Singleton Wrapper
class TrtManager {
public:
    // This function returns the SAME runtime instance every time
    static nvinfer1::IRuntime* getRuntime() {
        static TrtLogger logger; // Created once, lives forever
        
        // UPDATE: Changed from unique_ptr to raw pointer.
        // We intentionally "leak" this pointer at exit. 
        // If we try to delete it (via unique_ptr), the CUDA driver might 
        // already be closed, causing the "driver shutting down" error.
        static nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
        
        return runtime;
    }
};