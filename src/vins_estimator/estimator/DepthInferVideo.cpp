#include "vins_estimator/estimator/DepthInferVideo.h"
#include <fstream>
#include <iostream>
#include "vins_estimator/estimator/TrtCommon.h"

// Re-use logger or define static
static class VideoLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT-Video] " << msg << std::endl;
    }
} gVideoLogger;

DepthInferVideo::DepthInferVideo(std::string engine_path) {
    std::cout << "[DepthInferVideo] Loading Engine: " << engine_path << std::endl;

    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) { 
        std::cerr << "Error loading engine file!" << std::endl; 
        return; 
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    nvinfer1::IRuntime* runtime = TrtManager::getRuntime(); // Use existing manager if possible
    if (!runtime) runtime = nvinfer1::createInferRuntime(gVideoLogger); // Fallback

    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    delete[] trtModelStream;
    
    if (!engine) { std::cerr << "Failed to deserialize Engine!" << std::endl; return; }

    context = engine->createExecutionContext();
    
    // Print Bindings
    int nbBindings = engine->getNbIOTensors();
    std::cout << "[DepthInferVideo] Bindings: " << nbBindings << std::endl;
    for(int i=0; i<nbBindings; i++) {
        nvinfer1::Dims dims = engine->getTensorShape(engine->getIOTensorName(i));
        std::cout << "Binding " << i << ": " << engine->getIOTensorName(i) 
                  << " | Mode: " << (int)engine->getTensorIOMode(engine->getIOTensorName(i)) 
                  << " | Dims: [";
        for(int d=0; d<dims.nbDims; d++) std::cout << dims.d[d] << (d<dims.nbDims-1?"x":"");
        std::cout << "]" << std::endl;
    }

    // Allocate Image/Depth buffers
    // Assume Image is binding 'image' and Depth is 'depth'
    cudaMalloc((void**)&buffers[0], INPUT_SIZE * sizeof(float)); 
    cudaMalloc((void**)&buffers[1], OUTPUT_SIZE * sizeof(float)); 
    
    // Allocate State Input Buffers
    for (int i = 0; i < NUM_STATES; ++i) {
        size_t size_bytes = STATE_CONFIGS[i].size_bytes() * CONTEXT_LEN;
        cudaMalloc((void**)&state_input_buffers[i], size_bytes);
        cudaMemset(state_input_buffers[i], 0, size_bytes);
    }
    
    cudaError_t err = cudaMallocHost((void**)&h_input_pinned, INPUT_SIZE * sizeof(float));
    if (err != cudaSuccess) {
         std::cerr << "[DepthInferVideo] FATAL: cudaMallocHost failed! " << cudaGetErrorString(err) << std::endl;
         h_input_pinned = nullptr;
    }
    cudaStreamCreate(&stream);
    cpu_output_buffer = new float[OUTPUT_SIZE];

    // Start Worker
    running = true;
    worker_thread = std::thread(&DepthInferVideo::worker_loop, this);
}

DepthInferVideo::~DepthInferVideo() {
    running = false;
    input_cv.notify_all();
    if (worker_thread.joinable()) worker_thread.join();

    if(buffers[0]) cudaFree(buffers[0]);
    if(buffers[1]) cudaFree(buffers[1]);
    
    for(int i=0; i<NUM_STATES; ++i) {
        cudaFree(state_input_buffers[i]);
        for(auto p : history[i]) cudaFree(p);
        for(auto p : pool[i]) cudaFree(p);
    }
    
    if(h_input_pinned) cudaFreeHost(h_input_pinned);
    delete[] cpu_output_buffer;
    
    if(context) delete context;
    if(engine) delete engine;
    // runtime managed globally or deleted if created locally
}

float* DepthInferVideo::allocate_state(int idx) {
    if (!pool[idx].empty()) {
        float* p = pool[idx].back();
        pool[idx].pop_back();
        return p;
    }
    float* p = nullptr;
    cudaMalloc((void**)&p, STATE_CONFIGS[idx].size_bytes());
    return p;
}

void DepthInferVideo::release_state(int idx, float* p) {
    pool[idx].push_back(p);
}

void DepthInferVideo::input_image(double t, const cv::Mat& img) {
    std::lock_guard<std::mutex> lock(input_mutex);
    if (!running) return;
    
    // [LOKI] Latency Control: Drop oldest frames if queue gets too deep
    while (input_queue.size() >= 2) {
        double dropped_t = input_queue.front().first;
        input_queue.pop_front();
        // std::cout << "[Async] Queue Full. Dropping old frame " << std::fixed << dropped_t << " to catch up." << std::endl;
    }

    std::cout << "[Async] Pushing Input " << std::fixed << t << std::endl;
    input_queue.push_back({t, img.clone()}); // Clone essential for async safety
    input_cv.notify_one();
}

cv::Mat DepthInferVideo::get_depth(double t, int timeout_ms) {
    std::unique_lock<std::mutex> lock(output_mutex);
    // Wait up to timeout_ms
    if (output_map.find(t) == output_map.end()) {
        output_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, t]{ 
            return output_map.find(t) != output_map.end() || !running; 
        });
    }
    
    if (output_map.count(t)) {
        cv::Mat res = output_map[t];
        output_map.erase(t); // Consume result to keep map clean
        return res;
    }
    return cv::Mat(); // Return empty if timeout or not found
}

void DepthInferVideo::get_completed_depths(std::map<double, cv::Mat>& out_map) {
    std::lock_guard<std::mutex> lock(output_mutex);
    if (output_map.empty()) return;
    
    // std::cout << "[Async] Harvesting " << output_map.size() << " depths." << std::endl;
    
    // We want to TRANSFER all completed depths to the caller efficiently
    if (out_map.empty()) {
        output_map.swap(out_map); 
    } else {
        out_map.insert(output_map.begin(), output_map.end());
        output_map.clear();
    }
}

void DepthInferVideo::worker_loop() {
    while (running) {
        std::pair<double, cv::Mat> task;
        {
            std::unique_lock<std::mutex> lock(input_mutex);
            input_cv.wait(lock, [this]{ return !input_queue.empty() || !running; });
            if (!running && input_queue.empty()) break;
            task = input_queue.front();
            input_queue.pop_front();
        }
        
        std::cout << "[Async] Inferring " << std::fixed << task.first 
                  << " | Input size: " << task.second.cols << "x" << task.second.rows << std::endl;
        cv::Mat result = infer_internal(task.second);
        std::cout << "[Async] Done Inferring " << std::fixed << task.first 
                  << " | Output size: " << result.cols << "x" << result.rows 
                  << " | Empty: " << result.empty() << std::endl;
        
        // Optimization: Resize to original resolution HERE (Worker Thread)
        // This saves the main thread from doing it during Harvest
        cv::Mat resized_result;
        if (!result.empty()) {
            cv::resize(result, resized_result, task.second.size());
            std::cout << "[Async] Resized to: " << resized_result.cols << "x" << resized_result.rows << std::endl;
        } else {
            std::cout << "[Async] WARNING: Inference returned EMPTY depth for frame " << task.first << std::endl;
        }

        {
            std::lock_guard<std::mutex> lock(output_mutex);
            output_map[task.first] = resized_result;
        }
        output_cv.notify_all();
    }
}

cv::Mat DepthInferVideo::infer(cv::Mat& img) {
    return infer_internal(img);
}

cv::Mat DepthInferVideo::infer_internal(cv::Mat& img) {
    std::cout << "[DepthInferVideo] Checkpoint 1: Infer called. Input: " 
              << img.cols << "x" << img.rows << " channels=" << img.channels() << std::endl;
    if(!context) return cv::Mat();
    if(!h_input_pinned) {
        std::cerr << "[DepthInferVideo] ERROR: h_input_pinned is NULL" << std::endl;
        return cv::Mat();
    }

    // Preprocessing
    cv::Mat input_bgr;
    if (img.channels() == 1) {
        cv::cvtColor(img, input_bgr, cv::COLOR_GRAY2BGR);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, input_bgr, cv::COLOR_BGRA2BGR);
    } else {
        input_bgr = img;
    }

    cv::Mat resized;
    cv::resize(input_bgr, resized, cv::Size(INPUT_W, INPUT_H));
    std::cout << "[DepthInferVideo] Checkpoint 2: Resize done." << std::endl;

    const float mean_r = 0.485f; const float std_r = 0.229f;
    const float s_r = 1.0f / (255.0f * std_r); const float o_r = mean_r / std_r;
    const float mean_g = 0.456f; const float std_g = 0.224f;
    const float s_g = 1.0f / (255.0f * std_g); const float o_g = mean_g / std_g;
    const float mean_b = 0.406f; const float std_b = 0.225f;
    const float s_b = 1.0f / (255.0f * std_b); const float o_b = mean_b / std_b;

    float* p_r = h_input_pinned;
    float* p_g = h_input_pinned + (INPUT_W * INPUT_H);
    float* p_b = h_input_pinned + (2 * INPUT_W * INPUT_H);

    int total = INPUT_W * INPUT_H;
    const uchar* ptr = resized.ptr<uchar>(0);
    // BGR -> RGB & Norm
    std::cout << "[DepthInferVideo] Checkpoint 3: Starting normalization loop." << std::endl;
    for(int i=0; i<total; ++i) {
        p_r[i] = (static_cast<float>(ptr[3*i+2]) * s_r) - o_r;
        p_g[i] = (static_cast<float>(ptr[3*i+1]) * s_g) - o_g;
        p_b[i] = (static_cast<float>(ptr[3*i])   * s_b) - o_b;
    }
    std::cout << "[DepthInferVideo] Checkpoint 4: Loop done. Uploading..." << std::endl;

    cudaMemcpyAsync(buffers[0], h_input_pinned, INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice, stream);
    std::cout << "[DepthInferVideo] Checkpoint 5: Upload Async called. Processing History..." << std::endl;
    
    // --- State Construction ---
    indices_cache.clear();
    // Logic: 0, 1, ..., last 29
    int n = history[0].size();
    std::cout << "[DepthInferVideo] Checkpoint 5b: History size n=" << n << std::endl;
    if (n > 0) {
        if (n >= 1) indices_cache.push_back(0);
        if (n >= 2) indices_cache.push_back(1);
        int needed = CONTEXT_LEN - indices_cache.size();
        int start = std::max(0, n - needed);
        // Ensure strictly increasing from start?
        // Actually we just need last 'needed' frames.
        // If overlap with 0,1, it's fine (duplicated info?). 
        // Python logic: `cur_list = frame_cache_list[0:2] + frame_cache_list[-29:]`.
        // If list is small, say 5 elements: 0,1,2,3,4.
        // 0:2 -> 0,1.
        // -29: -> 0,1,2,3,4.
        // Result: 0,1,0,1,2,3,4.
        // This duplication seems intended or acceptable.
        
        for(int i = start; i < n; ++i) indices_cache.push_back(i);
        // Padding
        while(indices_cache.size() < CONTEXT_LEN) {
             indices_cache.push_back(n-1 >= 0 ? n-1 : 0);
        }
        
        // Strided Copy
        for(int s=0; s<NUM_STATES; ++s) {
            float* dst_base = (float*)state_input_buffers[s];
            int tokens = STATE_CONFIGS[s].tokens;
            int channels = STATE_CONFIGS[s].channels;
            size_t bytes_per_channel = channels * sizeof(float);
            
            for(int t=0; t<CONTEXT_LEN; ++t) {
                int frame = indices_cache[t];
                float* src = history[s][frame];
                
                // Copy logic: dst[token][t][channel] = src[token][channel]
                // Wait, TRT layout is [Tokens, 31, Channels] ?
                // If TRT layout is [Tokens, 31, Channels]:
                //   Stride(Tokens) = 31 * Channels
                //   Stride(Time) = Channels
                //   Stride(Channel) = 1
                // We want to copy into 't' slice.
                // It is NOT a contiguous block for 't'. It is scattered.
                // Access(token, t, ch) = token*31*C + t*C + ch
                // This means 't' slice blocks are separated by 31*C floats.
                // We have 'Tokens' such blocks.
                
                // Use cudaMemcpy2D
                // Width = Channels * sizeof(float) (One token's data for this time)
                // Height = Tokens
                // Src Pitch = Channels * sizeof(float)
                // Dst Pitch = 31 * Channels * sizeof(float)
                
                size_t width_bytes = channels * sizeof(float);
                size_t src_pitch = width_bytes;
                size_t dst_pitch = CONTEXT_LEN * width_bytes;
                
                // Dst offset: 0*31*C + t*C + 0 = t*C
                float* dst_ptr = dst_base + (t * channels);
                
                cudaMemcpy2DAsync(dst_ptr, dst_pitch, src, src_pitch, width_bytes, tokens, cudaMemcpyDeviceToDevice, stream);
            }
        }
    } else {
        // Init zero (already memset at init, but maybe set again if reset needed?)
        // Assuming history only empty at start.
    }
    
    // Bindings (V2 Style)
    // Indexes verified from logs:
    // 0: image
    // 1-8: state_in_0..7
    // 9: depth
    // 10-17: state_out_0..7
    
    void* bindings[18];
    bindings[0] = buffers[0]; // image
    bindings[9] = buffers[1]; // depth
    
    std::vector<float*> new_ptrs(NUM_STATES);
    for(int i=0; i<NUM_STATES; ++i) {
        bindings[1 + i] = state_input_buffers[i]; // state_in
        
        float* out = allocate_state(i);
        new_ptrs[i] = out;
        bindings[10 + i] = out; // state_out
    }
    
    // Execute
    std::cout << "[DepthInferVideo] Enqueueing V2... Stream=" << stream << std::endl;
    // context->enqueueV2(bindings, stream, nullptr);
    // Or executeV2 if synchronous debugging needed
    bool status = context->executeV2(bindings);
    
    if (!status) {
        std::cerr << "[DepthInferVideo] executeV2 Failed!" << std::endl;
        return cv::Mat();
    }
    std::cout << "[DepthInferVideo] Inference Done." << std::endl;
    
    cudaMemcpyAsync(cpu_output_buffer, buffers[1], OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::cout << "[DepthInferVideo] Inference Done." << std::endl;
    
    // Update History
    for(int i=0; i<NUM_STATES; ++i) {
        history[i].push_back(new_ptrs[i]);
        if(history[i].size() > 42) {
             // Keep 0,1. Remove 2.
             float* rem = history[i][2];
             history[i].erase(history[i].begin() + 2);
             release_state(i, rem);
        }
    }
    
    cv::Mat res(OUTPUT_H, OUTPUT_W, CV_32FC1, cpu_output_buffer);
    return res.clone();
}
