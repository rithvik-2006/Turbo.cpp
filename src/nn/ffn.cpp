// src/nn/ffn.cpp
#include "../../include/turbo/nn/ffn.hpp"

namespace turbo {
namespace nn {

FeedForward::FeedForward(size_t embed_dim, size_t intermediate_size)
    // We output 2x intermediate_size so SwiGLU can split it in half
    : gate_up_proj(embed_dim, intermediate_size * 2),
      down_proj(intermediate_size, embed_dim) {}

Tensor FeedForward::forward(const Tensor &input) {
  // 1. Expand and generate Gate & Up matrices simultaneously
  Tensor hidden = gate_up_proj.forward(input);

  // 2. Apply SwiGLU (Splits in half, activates, multiplies)
  hidden = swiglu.forward(hidden);

  // 3. Project back to embedding dimension
  return down_proj.forward(hidden);
}

} // namespace nn
} // namespace turbo