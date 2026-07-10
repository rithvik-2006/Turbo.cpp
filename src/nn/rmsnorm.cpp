// src/nn/rmsnorm.cpp
#include "../../include/turbo/nn/rmsnorm.hpp"
#include <cmath>
#include <stdexcept>

namespace turbo {
namespace nn {

RMSNorm::RMSNorm(size_t hidden_size, float epsilon)
    : hidden_size(hidden_size), epsilon(epsilon),
      weight(std::vector<float>(hidden_size, 0.0f), {hidden_size}) {

  // Initialize the gamma weight parameter to 1.0
  for (size_t i = 0; i < hidden_size; ++i) {
    weight.at({i}) = 1.0f;
  }
}

Tensor RMSNorm::forward(const Tensor &input) {
  auto in_shape = input.get_shape();
  if (in_shape.empty() || in_shape.back() != hidden_size) {
    throw std::invalid_argument("Input last dimension must match hidden_size.");
  }

  size_t numel = 1;
  for (auto d : in_shape) numel *= d;
  std::vector<float> zeros(numel, 0.0f);
  Tensor output(zeros, in_shape);

  // Calculate total number of rows (flattening all but the last dimension)
  size_t rows = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    rows *= in_shape[i];
  }

  // Process each row independently
  for (size_t i = 0; i < rows; ++i) {
    // 1. Calculate Sum of Squares
    float sum_of_squares = 0.0f;
    for (size_t j = 0; j < hidden_size; ++j) {
      float val = input.at({i, j});
      sum_of_squares += val * val;
    }

    // 2. Calculate the Root Mean Square (RMS)
    float mean_sq = sum_of_squares / hidden_size;
    float rms = std::sqrt(mean_sq + epsilon);

    // 3. Normalize and apply the learned weight vector
    for (size_t j = 0; j < hidden_size; ++j) {
      output.at({i, j}) = (input.at({i, j}) / rms) * weight.at({j});
    }
  }

  return output;
}

} // namespace nn
} // namespace turbo