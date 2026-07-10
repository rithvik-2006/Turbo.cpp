#include "../../include/turbo/nn/softmax.hpp"
#include "../../include/turbo/tensor/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
using namespace std;
namespace turbo {
namespace nn {

Tensor Softmax::forward(const Tensor &input) {
  std::vector<float> zeros(input.numel(), 0.0f);
  Tensor output(zeros, input.get_shape());
  size_t rows = input.get_shape()[0];
  size_t cols = input.get_shape()[1];

  for (size_t i = 0; i < rows; ++i) {
    float max_val = -INFINITY;
    for (size_t j = 0; j < cols; ++j) {
      float val = input.at({i, j});
      if (val > max_val) {
        max_val = val;
      }
    }
    float sum_exp = 0.0f;
    vector<float> exps(cols);
    for (size_t j = 0; j < cols; ++j) {
      exps[j] = exp(input.at({i, j}) - max_val);
      sum_exp += exps[j];
    }
    for (size_t j = 0; j < cols; ++j) {
      output.at({i, j}) = exps[j] / sum_exp;
    }
  }
  return output;
}
} // namespace nn
} // namespace turbo