#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>
// Uncomment these lines in benchmark_gemm.cpp:
#include "../include/turbo/tensor/tensor.hpp"

// Inside run_benchmark():

using namespace std;
using namespace std::chrono;

void run_benchmark(size_t M, size_t K, size_t N, int num_iterations = 10) {
  cout << "Benchmarking GEMM: " << M << "x" << K << " * " << K << "x" << N
       << "\n";

  // Allocate dummy data
  vector<float> A(M * K, 1.0f);
  vector<float> B(K * N, 2.0f);
  vector<float> C(M * N, 0.0f);

  // Warmup run (gets data into caches, spins up thread pool)
  matmul_threaded_packed_avx2(A.data(), B.data(), C.data(), M, K, N, K, N, N);

  double total_time_sec = 0.0;

  for (int i = 0; i < num_iterations; ++i) {
    auto start = high_resolution_clock::now();

    // Execute your highly optimized Stage 5 function here
    matmul_threaded_packed_avx2(A.data(), B.data(), C.data(), M, K, N, K, N, N);

    auto end = high_resolution_clock::now();
    total_time_sec += duration_cast<duration<double>>(end - start).count();
  }

  double avg_time_sec = total_time_sec / num_iterations;

  // Calculate Metrics
  double flops = 2.0 * M * N * K;
  double gflops = flops / (avg_time_sec * 1e9);

  double bytes_transferred = 4.0 * (M * K + K * N + M * N);
  double bandwidth_gbps = bytes_transferred / (avg_time_sec * 1e9);

  // At the bottom of run_benchmark(), right before it prints metrics:
  // This forces the compiler to keep the math loop alive!
  float dummy_sink = C[0];
  if (dummy_sink == 9999.0f) {
    cout << "This will never print, but prevents optimization: " << dummy_sink
         << "\n";
  }
  cout << fixed << setprecision(3);
  cout << "Average Latency : " << avg_time_sec * 1000.0 << " ms\n";
  cout << "Compute         : " << gflops << " GFLOPS\n";
  cout << "Bandwidth       : " << bandwidth_gbps << " GB/s\n";
  cout << "--------------------------------------------------\n";
}

int main() {
  cout << "--- Turbo Engine Benchmark Suite ---\n\n";

  // Test various sizes (L1 cache friendly -> RAM bound)
  run_benchmark(256, 256, 256);
  run_benchmark(512, 512, 512);
  run_benchmark(1024, 1024, 1024);
  run_benchmark(2048, 2048, 2048);

  return 0;
}