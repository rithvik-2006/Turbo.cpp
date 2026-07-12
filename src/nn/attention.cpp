// src/nn/attention.cpp
#include "../../include/turbo/nn/attention.hpp"
#include "../../include/turbo/core/kv_cache.hpp"
#include <cmath>
#include <stdexcept>
#include <cstring>

namespace turbo {
namespace nn {

MultiHeadAttention::MultiHeadAttention(Tensor q, Tensor k, Tensor v, Tensor o,
                                       size_t embed_dim, size_t num_heads, size_t num_kv_heads,
                                       size_t max_seq_len)
    : embed_dim(embed_dim), num_heads(num_heads), num_kv_heads(num_kv_heads),
      head_dim(embed_dim / num_heads), q_proj(std::move(q)),
      k_proj(std::move(k)), v_proj(std::move(v)),
      o_proj(std::move(o)),
      rope(embed_dim / num_heads, max_seq_len),
      softmax(-1)
{
  if (embed_dim % num_heads != 0) {
    throw std::invalid_argument(
        "Embedding dimension must be divisible by number of heads.");
  }
}

MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads, size_t num_kv_heads,
                                       size_t max_seq_len)
    : embed_dim(embed_dim), num_heads(num_heads), num_kv_heads(num_kv_heads),
      head_dim(embed_dim / num_heads), q_proj(embed_dim, embed_dim),
      k_proj(embed_dim, embed_dim), v_proj(embed_dim, embed_dim),
      o_proj(embed_dim, embed_dim),
      rope(embed_dim / num_heads, max_seq_len), // RoPE operates on head_dim
      softmax(-1) // Apply softmax over the last dimension (seq_len)
{
  if (embed_dim % num_heads != 0) {
    throw std::invalid_argument(
        "Embedding dimension must be divisible by number of heads.");
  }
}

Tensor MultiHeadAttention::forward(const Tensor &input,
                                   size_t position_offset,
                                   KVCache* kv_cache,
                                   int layer_idx) {
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
    
    Tensor& master_K_cache = kv_cache->get_k_cache(layer_idx);
    Tensor& master_V_cache = kv_cache->get_v_cache(layer_idx);

    // master_K_cache shape: [max_seq_len, num_heads, head_dim]
    // Slice a write-view into the exact position (dim=0 is seq_len)
    Tensor K_target_view = master_K_cache.slice(0, start_idx, start_idx + seq_len);
    Tensor V_target_view = master_V_cache.slice(0, start_idx, start_idx + seq_len);

    // Execute zero-copy in-place array assignment utilizing the new strided logic!
    K_target_view.copy_(K.squeeze(0).transpose(0, 1)); // We must transpose K back to [seq_len, num_heads, head_dim] for the copy
    V_target_view.copy_(V.squeeze(0).transpose(0, 1)); 

    // Now, slice the full contextual history and match the Q layout: [1, num_heads, history+seq, head_dim]
    K = master_K_cache.slice(0, 0, start_idx + seq_len).transpose(0, 1).unsqueeze(0);
    V = master_V_cache.slice(0, 0, start_idx + seq_len).transpose(0, 1).unsqueeze(0);
  }

  // 3. Scaled Dot-Product Attention
  // Support GQA: Repeat K and V heads to match num_heads
  if (num_heads != num_kv_heads) {
    size_t repeats = num_heads / num_kv_heads;
    size_t current_seq_len = K.get_shape()[2];
    std::vector<float> k_rep_data(batch_size * num_heads * current_seq_len * head_dim);
    std::vector<float> v_rep_data(batch_size * num_heads * current_seq_len * head_dim);
    Tensor K_rep(k_rep_data, {batch_size, num_heads, current_seq_len, head_dim});
    Tensor V_rep(v_rep_data, {batch_size, num_heads, current_seq_len, head_dim});
    
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
        for (size_t r = 0; r < repeats; ++r) {
          size_t h = kv_h * repeats + r;
          for (size_t s = 0; s < current_seq_len; ++s) {
            for (size_t d = 0; d < head_dim; ++d) {
              K_rep.at({b, h, s, d}) = K.at({b, kv_h, s, d});
              V_rep.at({b, h, s, d}) = V.at({b, kv_h, s, d});
            }
          }
        }
      }
    }
    K = K_rep;
    V = V_rep;
  }

  Tensor K_T = K.transpose(2, 3); 
  Tensor scores = Q.matmul(K_T); 

  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  scores = scores * scale;

  // Causal Masking (prefill phase)
  if (scores.get_shape().back() > 1 && seq_len > 1) {
    size_t cache_seq_len = scores.get_shape().back();
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t q_pos = 0; q_pos < seq_len; ++q_pos) {
        for (size_t k_pos = 0; k_pos < cache_seq_len; ++k_pos) {
          if (k_pos > q_pos + position_offset) {
            scores.at({0, h, q_pos, k_pos}) = -1e9f;
          }
        }
      }
    }
  }

  Tensor probs = softmax.forward(scores);
  Tensor context = probs.matmul(V); 

  // Reassemble Heads
  context = context.transpose(1, 2).contiguous().reshape({batch_size, seq_len, embed_dim});

  return o_proj.forward(context);
}

Tensor MultiHeadAttention::forward(const Tensor &input) {
  return forward(input, 0, nullptr, -1);
}

} // namespace nn
} // namespace turbo