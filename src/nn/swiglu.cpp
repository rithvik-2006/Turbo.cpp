// src/nn/swiglu.cpp
#include "../../include/turbo/nn/swiglu.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace turbo {
namespace nn {

Tensor SwiGLU::forward(const Tensor &gate, const Tensor &up) {
  auto gate_shape = gate.get_shape();
  auto up_shape = up.get_shape();

  if (gate_shape != up_shape) {
    throw std::invalid_argument("SwiGLU requires gate and up tensors to have the same shape.");
  }

  // Allocate the output tensor
  size_t numel = gate.numel();
  std::vector<float> zeros(numel, 0.0f);
  Tensor output(zeros, gate_shape);

  const float* gate_ptr = gate.data_ptr();
  const float* up_ptr = up.data_ptr();
  float* out_ptr = output.data_ptr();

  // 2. Element-wise C++ Loop for SwiGLU
  for (size_t i = 0; i < numel; ++i) {
    float x1 = gate_ptr[i];
    float x2 = up_ptr[i];

    // Apply SiLU to X1: x1 / (1.0 + exp(-x1))
    float silu_x1 = x1 / (1.0f + std::exp(-x1));

    // Multiply by X2
    out_ptr[i] = silu_x1 * x2;
  }

  return output;
}

Tensor SwiGLU::forward(const Tensor &input) {
  throw std::runtime_error("SwiGLU::forward(const Tensor &input) is deprecated. Use forward(gate, up).");
}

} // namespace nn
} // namespace turbo