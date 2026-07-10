// include/turbo/nn/ffn.hpp
#pragma once
#include "layer.hpp"
#include "linear.hpp"
#include "swiglu.hpp"

namespace turbo {
namespace nn {

class FeedForward : public Layer {
private:
  Linear gate_up_proj;
  SwiGLU swiglu;
  Linear down_proj;

public:
  // intermediate_size is typically 4x the embed_dim in standard models,
  // but around 8/3x in LLaMA models.
  FeedForward(size_t embed_dim, size_t intermediate_size);

  Tensor forward(const Tensor &input) override;
};

} // namespace nn
} // namespace turbo