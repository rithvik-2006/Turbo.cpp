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

  std::vector<float> zeros(input.numel(), 0.0f);
  Tensor output(zeros, input.get_shape());
  
  size_t cols = input.get_shape().back();
  size_t rows = input.numel() / cols;

  const float* in_ptr = input.data_ptr();
  float* out_ptr = output.data_ptr();

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