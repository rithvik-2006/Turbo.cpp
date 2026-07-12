// include/turbo/nn/transformer_block.hpp
#pragma once
#include "attention.hpp"
#include "ffn.hpp"
#include "layer.hpp"
#include "rmsnorm.hpp"

namespace turbo {
class KVCache;
class GGUFLoader;
namespace nn {

class TransformerBlock : public Layer {
private:
  RMSNorm attention_norm;
  MultiHeadAttention attention;

  RMSNorm ffn_norm;
  FeedForward ffn;

public:
  // Zero copy constructor from GGUF loader
  TransformerBlock(GGUFLoader& loader, int layer_idx, size_t embed_dim, size_t num_heads, size_t num_kv_heads, size_t max_seq_len = 4096);

  TransformerBlock(size_t embed_dim, size_t num_heads, size_t num_kv_heads, size_t intermediate_size,
                   size_t max_seq_len = 4096);

  // Forward pass requires position offset for RoPE tracking
  Tensor forward(const Tensor &input, size_t position_offset, KVCache* kv_cache = nullptr, int layer_idx = -1);

  // Fallback for standard Layer interface
  Tensor forward(const Tensor &input) override;
};

} // namespace nn
} // namespace turbo