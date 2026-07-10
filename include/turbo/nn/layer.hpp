#pragma once
#include "../tensor/tensor.hpp"

namespace turbo {
namespace nn {
class Layer {
public:
  virtual ~Layer() = default;
  virtual Tensor forward(const Tensor &input) = 0;
};
} // namespace nn
} // namespace turbo
