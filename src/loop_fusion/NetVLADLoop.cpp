#include "loop_fusion/NetVLADLoop.h"
#include <fstream>
#include <algorithm>

NetVLADLoop::NetVLADLoop(const std::string& engine_path, int max_db_size) 
    : MAX_DB_SIZE(max_db_size) 
{
    // 1. Load TensorRT Engine
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[Error] Engine file not found: " << engine_path << std::endl;
        return;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    context = engine->createExecutionContext();
    delete[] trtModelStream;

    // 2. Initialize cuBLAS
    cublasCreate(&cublas_handle);

    // 3. Allocate GPU Memory
    size_t input_bytes = 1 * 3 * INPUT_H * INPUT_W * sizeof(float);
    size_t desc_bytes = DESC_DIM * sizeof(float);
    size_t db_bytes = (size_t)MAX_DB_SIZE * DESC_DIM * sizeof(float);
    size_t score_bytes = MAX_DB_SIZE * sizeof(float);

    cudaMalloc(&buffers[0], input_bytes);     // Input Image
    cudaMalloc(&buffers[1], desc_bytes);      // Current Descriptor (TRT Output)
    cudaMalloc((void**)&d_database, db_bytes);
    cudaMalloc((void**)&d_scores, score_bytes);

    // 4. Allocate CPU Memory
    h_scores = new float[MAX_DB_SIZE];
}

NetVLADLoop::~NetVLADLoop() {
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(d_database);
    cudaFree(d_scores);
    delete[] h_scores;
    
    cublasDestroy(cublas_handle);
    delete context;
    delete engine;
    delete runtime;
}

std::vector<float> NetVLADLoop::extract_and_add(const cv::Mat& img, int frame_index) {
    // --- 1. PREPROCESS (CPU) ---
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
    cv::Mat rgb_img;
    cv::cvtColor(resized, rgb_img, cv::COLOR_BGR2RGB);
    cv::Mat float_img;
    rgb_img.convertTo(float_img , CV_32FC3);
    

    std::vector<float> input_data;
    input_data.reserve(3 * INPUT_H * INPUT_W);
    std::vector<cv::Mat> channels;
    cv::split(float_img, channels);
    for (auto& ch : channels) 
        input_data.insert(input_data.end(), (float*)ch.data, (float*)ch.data + ch.total());

    // --- 2. INFERENCE (GPU) ---
    cudaMemcpy(buffers[0], input_data.data(), input_data.size()*sizeof(float), cudaMemcpyHostToDevice);
    context->executeV2(buffers);

    // --- 3. ADD TO GPU DB ---
    if (current_db_size < MAX_DB_SIZE) {
        // Copy the new descriptor from TRT output buffer -> Main Database Buffer
        // Offset = current_db_size * DESC_DIM
        float* db_ptr = d_database + (size_t)current_db_size * DESC_DIM;
        cudaMemcpy(db_ptr, buffers[1], DESC_DIM * sizeof(float), cudaMemcpyDeviceToDevice);
        
        db_indices.push_back(frame_index);
        current_db_size++;
    }

    // --- 4. RETURN TO CPU (For saving to disk) ---
    std::vector<float> cpu_desc(DESC_DIM);
    cudaMemcpy(cpu_desc.data(), buffers[1], DESC_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    return cpu_desc;
}

std::pair<int, float> NetVLADLoop::query(int min_interval) {
    if (current_db_size <= min_interval) return {-1, 0.0f};

    // --- GPU SEARCH (Matrix-Vector Multiplication) ---
    // Operation: y = alpha * A * x + beta * y
    // A (Database) is [DESC_DIM x N] (Stored Column-Major effectively if we treat descriptors as columns)
    // Actually, cuBLAS assumes Column-Major. Our data is Row-Major (N rows of Descriptors).
    // So A is [N x DESC_DIM]. We want A * x (dot product of every row with x).
    // In cuBLAS logic, Row-Major A is Transposed Column-Major.
    // So we tell cuBLAS: "Treat A as Transposed", dimensions are DESC_DIM x current_db_size.
    
    float alpha = 1.0f;
    float beta = 0.0f;
    int m = current_db_size; // Number of rows (descriptors)
    int n = DESC_DIM;        // Dimension
    
    // Check 'cublasSgemv' docs: y = alpha * op(A) * x + beta * y
    // We pass CUBLAS_OP_T because our layout in memory is Row-Major (C++ default), 
    // but cuBLAS expects Col-Major.
    
    cublasSgemv(cublas_handle, CUBLAS_OP_T, 
                n, m, 
                &alpha, 
                d_database, n, // Leading dimension is DESC_DIM
                (float*)buffers[1], 1, // The Query (Current Descriptor)
                &beta, 
                d_scores, 1);

    // --- READ RESULTS ---
    // Copy scores back to CPU to find Max
    cudaMemcpy(h_scores, d_scores, m * sizeof(float), cudaMemcpyDeviceToHost);

    // --- FIND MAX ---
    int best_idx = -1;
    float best_score = 0.0f;

    // We search up to (current - min_interval)
    int search_limit = current_db_size - min_interval;
    
    for (int i = 0; i < search_limit; ++i) {
        if (h_scores[i] > best_score) {
            best_score = h_scores[i];
            best_idx = db_indices[i];
        }
    }

    return {best_idx, best_score};
}