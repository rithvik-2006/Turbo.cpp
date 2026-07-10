// src/nn/linear.cpp
#include "../../include/turbo/nn/linear.hpp"
#include <vector>

namespace turbo {
namespace nn {

Linear::Linear(size_t in_features, size_t out_features)
    : in_features(in_features), out_features(out_features),
      // Initialize weight tensor: Shape [in_features, out_features]
      weight(std::vector<float>(in_features * out_features, 0.0f), {in_features, out_features}),
      // Initialize bias tensor: Shape [out_features]
      bias(std::vector<float>(out_features, 0.0f), {out_features}) {

  // In a real inference engine, these would be populated by your GGUF Loader.
  // For now, we will zero-initialize them (or you could add a random init
  // loop). The Tensor constructor already allocates the memory securely.
}

Tensor Linear::forward(const Tensor &input) {
  // 1. Matrix Multiplication: XW
  // This routes directly to your Stage 5 Threaded, Cache-Packed GEMM Engine
  Tensor output = input.matmul(weight);

  // 2. Add Bias: + b
  // This routes to your Project 1 overloaded operator+, triggering
  // automatic zero-stride broadcasting for the 1D bias vector.
  output = output + bias;

  return output;
}

} // namespace nn
} // namespace turbo