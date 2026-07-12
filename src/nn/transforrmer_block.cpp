// src/nn/transformer_block.cpp
#include "../../include/turbo/nn/transformer_block.hpp"
#include "../../include/turbo/loader/gguf_loader.hpp"
#include <string>

namespace turbo {
namespace nn {

TransformerBlock::TransformerBlock(GGUFLoader& loader, int layer_idx, size_t embed_dim, size_t num_heads, size_t num_kv_heads, size_t max_seq_len)
    : attention_norm(loader.get_tensor("blk." + std::to_string(layer_idx) + ".attn_norm.weight")),
      attention(
          loader.get_tensor("blk." + std::to_string(layer_idx) + ".attn_q.weight"),
          loader.get_tensor("blk." + std::to_string(layer_idx) + ".attn_k.weight"),
          loader.get_tensor("blk." + std::to_string(layer_idx) + ".attn_v.weight"),
          loader.get_tensor("blk." + std::to_string(layer_idx) + ".attn_output.weight"),
          embed_dim, num_heads, num_kv_heads, max_seq_len
      ),
      ffn_norm(loader.get_tensor("blk." + std::to_string(layer_idx) + ".ffn_norm.weight")),
      ffn(
          loader.get_tensor("blk." + std::to_string(layer_idx) + ".ffn_gate.weight"),
          loader.get_tensor("blk." + std::to_string(layer_idx) + ".ffn_down.weight"),
          loader.get_tensor("blk." + std::to_string(layer_idx) + ".ffn_up.weight")
      ) {}

TransformerBlock::TransformerBlock(size_t embed_dim, size_t num_heads, size_t num_kv_heads,
                                   size_t intermediate_size, size_t max_seq_len)
    : attention_norm(embed_dim), attention(embed_dim, num_heads, num_kv_heads, max_seq_len),
      ffn_norm(embed_dim), ffn(embed_dim, intermediate_size) {}

Tensor TransformerBlock::forward(const Tensor &input, size_t position_offset, KVCache* kv_cache, int layer_idx) {
  // 1. Attention Block (Pre-Norm + Residual)
  Tensor norm_x = attention_norm.forward(input);
  Tensor attn_out = attention.forward(norm_x, position_offset, kv_cache, layer_idx);

  // Uses your overloaded operator+ for zero-copy broadcasting addition
  Tensor h = input + attn_out;

  // 2. Feed-Forward Block (Pre-Norm + Residual)
  Tensor norm_h = ffn_norm.forward(h);
  Tensor ffn_out = ffn.forward(norm_h);

  Tensor out = h + ffn_out;

  return out;
}

Tensor TransformerBlock::forward(const Tensor &input) {
  return forward(input, 0);
}

} // namespace nn
} // namespace turbo