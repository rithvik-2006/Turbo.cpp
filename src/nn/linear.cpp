// src/nn/linear.cpp
#include "../../include/turbo/nn/linear.hpp"
#include <vector>

namespace turbo {
namespace nn {

Linear::Linear(Tensor w, Tensor b) 
    : weight(std::move(w)), bias(std::move(b)), 
      in_features(weight.get_shape().size() > 0 ? weight.get_shape()[0] : 0),
      out_features(weight.get_shape().size() > 1 ? weight.get_shape()[1] : 0) {
}

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
  auto in_shape = input.get_shape();
  
  Tensor flattened_input = input;
  if (in_shape.size() > 2) {
    size_t rows = 1;
    for (size_t i = 0; i < in_shape.size() - 1; ++i) {
      rows *= in_shape[i];
    }
    // We assume the input is contiguous here for the fast reshape
    flattened_input = input.reshape({rows, in_shape.back()});
  }

  // 1. Matrix Multiplication: XW
  Tensor output = flattened_input.matmul(weight);

  // 2. Add Bias: + b (if present)
  if (!bias.is_empty()) {
      output = output + bias;
  }

  // Restore the original dimensional hierarchy if we flattened it
  if (in_shape.size() > 2) {
    std::vector<size_t> out_shape = in_shape;
    out_shape.back() = out_features;
    output = output.reshape(out_shape);
  }

  return output;
}

} // namespace nn
} // namespace turbo