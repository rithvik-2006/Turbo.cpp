// src/nn/swiglu.cpp
#include "../../include/turbo/nn/swiglu.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace turbo {
namespace nn {

Tensor SwiGLU::forward(const Tensor &input) {
  auto in_shape = input.get_shape();
  if (in_shape.empty() || in_shape.back() % 2 != 0) {
    throw std::invalid_argument(
        "SwiGLU requires the last dimension to be even for splitting.");
  }

  // 1. Calculate new shape (hidden dimension is halved)
  std::vector<size_t> out_shape = in_shape;
  size_t half_dim = in_shape.back() / 2;
  out_shape.back() = half_dim;

  // Allocate the output tensor
  size_t numel = 1;
  for (size_t dim : out_shape) numel *= dim;
  std::vector<float> zeros(numel, 0.0f);
  Tensor output(zeros, out_shape);

  size_t rows = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    rows *= in_shape[i];
  }

  // 2. Element-wise C++ Loop for SwiGLU
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < half_dim; ++j) {
      // Extract X1 (first half) and X2 (second half)
      float x1 = input.at({i, j});
      float x2 = input.at({i, j + half_dim});

      // Apply SiLU to X1: x1 / (1.0 + exp(-x1))
      float silu_x1 = x1 / (1.0f + std::exp(-x1));

      // Element-wise multiplication and write to output
      output.at({i, j}) = silu_x1 * x2;
    }
  }

  return output;
}

} // namespace nn
} // namespace turbo