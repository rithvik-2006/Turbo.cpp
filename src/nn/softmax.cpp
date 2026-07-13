// src/nn/softmax.cpp
#include "../../include/turbo/nn/softmax.hpp"
#include "../../include/turbo/tensor/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
using namespace std;
namespace turbo {
namespace nn {

Tensor Softmax::forward(const Tensor &input) {
  if (input.get_shape().empty()) {
    throw std::invalid_argument("Cannot softmax an empty tensor.");
  }

  // IMPORTANT: input may be a non-contiguous view (result of slice/transpose +
  // scalar multiply). Raw pointer arithmetic only works on contiguous memory.
  // Make contiguous first to ensure correct row-major element access.
  Tensor contig_input = input.contiguous();

  std::vector<float> zeros(contig_input.numel(), 0.0f);
  Tensor output(zeros, contig_input.get_shape());

  size_t cols = contig_input.get_shape().back();
  size_t rows = contig_input.numel() / cols;

  const float *in_ptr = contig_input.data_ptr();
  float *out_ptr = output.data_ptr();

  for (size_t i = 0; i < rows; ++i) {
    float max_val = -INFINITY;
    for (size_t j = 0; j < cols; ++j) {
      float val = in_ptr[i * cols + j];
      if (val > max_val) {
        max_val = val;
      }
    }

    float sum_exp = 0.0f;
    for (size_t j = 0; j < cols; ++j) {
      float e = exp(in_ptr[i * cols + j] - max_val);
      out_ptr[i * cols + j] = e;
      sum_exp += e;
    }

    for (size_t j = 0; j < cols; ++j) {
      out_ptr[i * cols + j] /= sum_exp;
    }
  }

  return output;
}
} // namespace nn
} // namespace turbo