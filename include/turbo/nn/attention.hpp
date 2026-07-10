// include/turbo/nn/attention.hpp
#pragma once
#include "layer.hpp"
#include "linear.hpp"
#include "rope.hpp"
#include "softmax.hpp"

namespace turbo {
namespace nn {

class MultiHeadAttention : public Layer {
private:
  size_t embed_dim;
  size_t num_heads;
  size_t head_dim;

  // Sub-components
  Linear q_proj;
  Linear k_proj;
  Linear v_proj;
  Linear o_proj;

  RoPE rope;
  Softmax softmax;

public:
  MultiHeadAttention(size_t embed_dim, size_t num_heads,
                     size_t max_seq_len = 4096);

  // Standard forward pass (for Prefill phase)
  Tensor forward(const Tensor &input) override;

  // Overloaded forward pass for Decode phase (requires position tracking for
  // RoPE)
  Tensor forward(const Tensor &input, size_t position_offset);
};

} // namespace nn
} // namespace turbo