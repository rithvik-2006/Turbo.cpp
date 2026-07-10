// include/turbo/nn/transformer_block.hpp
#pragma once
#include "attention.hpp"
#include "ffn.hpp"
#include "layer.hpp"
#include "rmsnorm.hpp"

namespace turbo {
namespace nn {

class TransformerBlock : public Layer {
private:
  RMSNorm attention_norm;
  MultiHeadAttention attention;

  RMSNorm ffn_norm;
  FeedForward ffn;

public:
  TransformerBlock(size_t embed_dim, size_t num_heads, size_t intermediate_size,
                   size_t max_seq_len = 4096);

  // Forward pass requires position offset for RoPE tracking
  Tensor forward(const Tensor &input, size_t position_offset);

  // Fallback for standard Layer interface
  Tensor forward(const Tensor &input) override;
};

} // namespace nn
} // namespace turbo