// src/nn/ffn.cpp
#include "../../include/turbo/nn/ffn.hpp"
#include <iostream>
#include <cmath>

namespace turbo {
namespace nn {

FeedForward::FeedForward(Tensor gate, Tensor down, Tensor up)
    : gate_proj(std::move(gate)), down_proj(std::move(down)), up_proj(std::move(up)) {}

FeedForward::FeedForward(size_t embed_dim, size_t intermediate_size)
    : gate_proj(embed_dim, intermediate_size),
      up_proj(embed_dim, intermediate_size),
      down_proj(intermediate_size, embed_dim) {}

Tensor FeedForward::forward(const Tensor &input) {
  // 1. Generate Gate & Up matrices
  Tensor gate = gate_proj.forward(input);
  Tensor up = up_proj.forward(input);

  // 2. Apply SwiGLU (activates gate, multiplies with up)
  Tensor hidden = swiglu.forward(gate, up);

  // 3. Project back to embedding dimension
  return down_proj.forward(hidden);
}

} // namespace nn
} // namespace turbo