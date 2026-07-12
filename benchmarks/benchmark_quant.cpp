#include "../include/turbo/tensor/tensor.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace turbo;

void run_gemm_benchmark() {
    int dim = 4096; // Standard LLaMA-7B hidden dimension
    
    // Create Dummy Tensors
    std::vector<float> W_fp32_data(dim * dim, 1.0f);
    Tensor W_fp32(W_fp32_data, {(size_t)dim, (size_t)dim});
    
    size_t q8_bytes = (dim * dim) / get_block_size(DataType::Q8_0) * get_type_size(DataType::Q8_0);
    std::vector<uint8_t> W_q8_data(q8_bytes, 0);
    Tensor W_q8({(size_t)dim, (size_t)dim}, DataType::Q8_0, W_q8_data.data());

    std::vector<float> X_data(dim, 1.0f);
    Tensor X(X_data, {1, (size_t)dim});

    std::cout << "--- Turbo.cpp GEMM Benchmark ---" << std::endl;
    std::cout << "Matrix Dimensions: [" << dim << "x" << dim << "]" << std::endl;

    // Warmup
    Tensor Y_fp32_warm = X.matmul(W_fp32);
    Tensor Y_q8_warm = X.matmul(W_q8);

    // 1. Profile FP32
    auto start_fp32 = std::chrono::high_resolution_clock::now();
    
    // Run multiple iterations to get stable metrics
    int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        Tensor Y_fp32 = X.matmul(W_fp32);
    }
    
    auto end_fp32 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_fp32 = end_fp32 - start_fp32;
    double avg_ms_fp32 = ms_fp32.count() / iterations;
    
    // FP32 Math: 4096 * 4096 elements * 4 bytes = ~67 MB moved
    double gb_moved_fp32 = (dim * dim * 4.0) / (1024 * 1024 * 1024);
    double bandwidth_fp32 = gb_moved_fp32 / (avg_ms_fp32 / 1000.0);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "FP32 Matmul: " << avg_ms_fp32 << " ms | Bandwidth: " 
              << bandwidth_fp32 << " GB/s" << std::endl;

    // 2. Profile Q8_0
    auto start_q8 = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        Tensor Y_q8 = X.matmul(W_q8);
    }
    
    auto end_q8 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_q8 = end_q8 - start_q8;
    double avg_ms_q8 = ms_q8.count() / iterations;
    
    // Q8_0 Math: 4096 * 4096 elements. Each block of 32 takes 34 bytes.
    double gb_moved_q8 = ((dim * dim) / 32.0 * 34.0) / (1024 * 1024 * 1024);
    double bandwidth_q8 = gb_moved_q8 / (avg_ms_q8 / 1000.0);
    
    std::cout << "Q8_0 Matmul: " << avg_ms_q8 << " ms | Bandwidth: " 
              << bandwidth_q8 << " GB/s" << std::endl;
              
    std::cout << "Q8_0 Speedup Factor: " << avg_ms_fp32 / avg_ms_q8 << "x\n" << std::endl;

    // 3. Profile Q4_0
    size_t q4_bytes = (dim * dim) / get_block_size(DataType::Q4_0) * get_type_size(DataType::Q4_0);
    std::vector<uint8_t> W_q4_data(q4_bytes, 0);
    Tensor W_q4({(size_t)dim, (size_t)dim}, DataType::Q4_0, W_q4_data.data());

    // Warmup Q4_0
    Tensor Y_q4_warm = X.matmul(W_q4);

    auto start_q4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        Tensor Y_q4 = X.matmul(W_q4);
    }
    auto end_q4 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> ms_q4 = end_q4 - start_q4;
    double avg_ms_q4 = ms_q4.count() / iterations;
    
    // Q4_0 Math: 4096 * 4096 elements. Each block of 32 takes 18 bytes.
    double gb_moved_q4 = ((dim * dim) / 32.0 * 18.0) / (1024 * 1024 * 1024);
    double bandwidth_q4 = gb_moved_q4 / (avg_ms_q4 / 1000.0);
    
    std::cout << "Q4_0 Matmul: " << avg_ms_q4 << " ms | Bandwidth: " 
              << bandwidth_q4 << " GB/s" << std::endl;
              
    std::cout << "Q4_0 Speedup Factor: " << avg_ms_fp32 / avg_ms_q4 << "x\n" << std::endl;
}

int main() {
    run_gemm_benchmark();
    return 0;
}
