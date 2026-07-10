// include/turbo/nn/swiglu.hpp
#pragma once
#include "layer.hpp"

namespace turbo {
namespace nn {

class SwiGLU : public Layer {
public:
  SwiGLU() = default;

  // The execution contract
  Tensor forward(const Tensor &input) override;
};

} // namespace nn
} // namespace turbo