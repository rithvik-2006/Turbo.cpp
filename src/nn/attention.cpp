// src/nn/attention.cpp
#include "../../include/turbo/nn/attention.hpp"
#include "../../include/turbo/core/kv_cache.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace turbo {
namespace nn {

MultiHeadAttention::MultiHeadAttention(Tensor q, Tensor k, Tensor v, Tensor o,
                                       size_t embed_dim, size_t num_heads,
                                       size_t num_kv_heads, size_t max_seq_len,
                                       float theta_base)
    : embed_dim(embed_dim), num_heads(num_heads), num_kv_heads(num_kv_heads),
      head_dim(embed_dim / num_heads), q_proj(std::move(q)),
      k_proj(std::move(k)), v_proj(std::move(v)), o_proj(std::move(o)),
      rope(embed_dim / num_heads, max_seq_len, theta_base), softmax(-1) {
  if (embed_dim % num_heads != 0) {
    throw std::invalid_argument(
        "Embedding dimension must be divisible by number of heads.");
  }
}

MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads,
                                       size_t num_kv_heads, size_t max_seq_len,
                                       float theta_base)
    : embed_dim(embed_dim), num_heads(num_heads), num_kv_heads(num_kv_heads),
      head_dim(embed_dim / num_heads), q_proj(embed_dim, embed_dim),
      k_proj(embed_dim, embed_dim), v_proj(embed_dim, embed_dim),
      o_proj(embed_dim, embed_dim),
      rope(embed_dim / num_heads, max_seq_len, theta_base), // RoPE operates on head_dim
      softmax(-1) // Apply softmax over the last dimension (seq_len)
{
  if (embed_dim % num_heads != 0) {
    throw std::invalid_argument(
        "Embedding dimension must be divisible by number of heads.");
  }
}

Tensor MultiHeadAttention::forward(const Tensor &input, size_t position_offset,
                                   KVCache *kv_cache, int layer_idx) {
  size_t batch_size = input.get_shape()[0];
  size_t seq_len = input.get_shape()[1];

  // 1. Projections
  Tensor Q = q_proj.forward(input);
  Tensor K = k_proj.forward(input);
  Tensor V = v_proj.forward(input);

  // Reshape to separate heads: [batch, seq_len, num_heads, head_dim]
  Q = Q.reshape({batch_size, seq_len, num_heads, head_dim});
  K = K.reshape({batch_size, seq_len, num_kv_heads, head_dim});
  V = V.reshape({batch_size, seq_len, num_kv_heads, head_dim});

  // RoPE expects [batch, num_heads, seq_len, head_dim]
  Q = Q.transpose(1, 2);
  K = K.transpose(1, 2);
  V = V.transpose(1, 2);

  // Apply RoPE
  rope.apply_in_place(Q, position_offset);
  rope.apply_in_place(K, position_offset);

  // 2. KV Cache Slicing Orchestration
  if (kv_cache != nullptr && layer_idx >= 0) {
    size_t start_idx = kv_cache->get_current_seq_len();

    Tensor &master_K_cache = kv_cache->get_k_cache(layer_idx);
    Tensor &master_V_cache = kv_cache->get_v_cache(layer_idx);

    // master_K_cache shape: [max_seq_len, num_heads, head_dim]
    // Slice a write-view into the exact position (dim=0 is seq_len)
    Tensor K_target_view =
        master_K_cache.slice(0, start_idx, start_idx + seq_len);
    Tensor V_target_view =
        master_V_cache.slice(0, start_idx, start_idx + seq_len);

    // Execute zero-copy in-place array assignment utilizing the new strided
    // logic!
    K_target_view.copy_(
        K.squeeze(0).transpose(0, 1)); // We must transpose K back to [seq_len,
                                       // num_heads, head_dim] for the copy
    V_target_view.copy_(V.squeeze(0).transpose(0, 1));

    // Now, slice the full contextual history and match the Q layout: [1,
    // num_heads, history+seq, head_dim]
    K = master_K_cache.slice(0, 0, start_idx + seq_len)
            .transpose(0, 1)
            .unsqueeze(0);
    V = master_V_cache.slice(0, 0, start_idx + seq_len)
            .transpose(0, 1)
            .unsqueeze(0);
  }

  // 3. Scaled Dot-Product Attention (Fix 1: Zero-Copy GQA loop)
  Tensor context = Tensor(
      std::vector<float>(batch_size * num_heads * seq_len * head_dim, 0.0f),
      {batch_size, num_heads, seq_len, head_dim});

  size_t n_rep = num_heads / num_kv_heads;
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      size_t kv_h =
          h / n_rep; // Heads MUST share the physical KV head mapped by n_rep!

      // Slice Q and K down to their explicit head dimensions
      Tensor Q_head = Q.slice(0, b).slice(0, h);
      Tensor K_head = K.slice(0, b).slice(0, kv_h);
      Tensor V_head = V.slice(0, b).slice(0, kv_h);

      // Dot product Q * K^T
      Tensor K_head_T = K_head.transpose(
          0, 1); // K_head is [seq_len, head_dim] -> [head_dim, seq_len]
      Tensor scores = Q_head.matmul(K_head_T) * scale;

      // Causal Masking (prefill phase)
      if (scores.get_shape().back() > 1 && seq_len > 1) {
        size_t cache_seq_len = scores.get_shape().back();
        for (size_t q_pos = 0; q_pos < seq_len; ++q_pos) {
          for (size_t k_pos = 0; k_pos < cache_seq_len; ++k_pos) {
            if (k_pos > q_pos + position_offset) {
              scores.at({q_pos, k_pos}) = -1e9f;
            }
          }
        }
      }

      Tensor probs_head = softmax.forward(scores);
      Tensor context_head = probs_head.matmul(V_head);

      // Write directly back to our context buffer
      for (size_t s = 0; s < seq_len; ++s) {
        for (size_t d = 0; d < head_dim; ++d) {
          context.at({b, h, s, d}) = context_head.at({s, d});
        }
      }
    }
  }

  // Reassemble Heads
  context = context.transpose(1, 2).contiguous().reshape(
      {batch_size, seq_len, embed_dim});

  return o_proj.forward(context);
}

Tensor MultiHeadAttention::forward(const Tensor &input) {
  return forward(input, 0, nullptr, -1);
}

} // namespace nn
} // namespace turbo