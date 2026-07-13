// include/turbo/nn/attention.hpp
#pragma once
#include "layer.hpp"
#include "linear.hpp"
#include "rope.hpp"
#include "softmax.hpp"

namespace turbo {

class KVCache;

namespace nn {

class MultiHeadAttention : public Layer {
private:
  size_t embed_dim;
  size_t num_heads;
  size_t num_kv_heads;
  size_t head_dim;

  // Sub-components
  Linear q_proj;
  Linear k_proj;
  Linear v_proj;
  Linear o_proj;

  RoPE rope;
  Softmax softmax;

public:
  // Zero-copy constructor
  MultiHeadAttention(Tensor q, Tensor k, Tensor v, Tensor o, size_t embed_dim,
                     size_t num_heads, size_t num_kv_heads,
                     size_t max_seq_len = 4096, float theta_base = 10000.0f);

  // Legacy constructor
  MultiHeadAttention(size_t embed_dim, size_t num_heads, size_t num_kv_heads,
                     size_t max_seq_len = 4096, float theta_base = 10000.0f);

  // Standard forward pass (for Prefill phase)
  Tensor forward(const Tensor &input) override;

  // Overloaded forward pass for Decode phase (requires position tracking for
  // RoPE and KVCache)
  Tensor forward(const Tensor &input, size_t position_offset,
                 KVCache *kv_cache = nullptr, int layer_idx = -1);
};

} // namespace nn
} // namespace turbo