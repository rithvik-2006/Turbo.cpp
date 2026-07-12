// include/turbo/nn/linear.hpp
#pragma once
#include "../tensor/tensor.hpp"
#include "layer.hpp"
#include <random>

namespace turbo {
namespace nn {

class Linear : public Layer {
private:
  Tensor weight;
  Tensor bias;
  size_t in_features;
  size_t out_features;

public:
  // Default constructor for lazy initialization
  Linear() = default;

  // Constructor from GGUF loaded tensors (Zero-copy)
  Linear(Tensor w, Tensor b = Tensor());

  // The constructor allocates the weight and bias tensors (Legacy)
  Linear(size_t in_features, size_t out_features);

  // The execution contract
  Tensor forward(const Tensor &input) override;

  // Getters for testing and model loading
  Tensor &get_weight() { return weight; }
  Tensor &get_bias() { return bias; }
};

} // namespace nn
} // namespace turbo