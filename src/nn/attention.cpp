// src/nn/attention.cpp
#include "../../include/turbo/nn/attention.hpp"
#include <cmath>
#include <stdexcept>

namespace turbo {
namespace nn {

MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads,
                                       size_t max_seq_len)
    : embed_dim(embed_dim), num_heads(num_heads),
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
                                   size_t position_offset) {
  // 1. Projections (Y = XW + b) -> Routes to your AVX2 GEMM
  Tensor Q = q_proj.forward(input);
  Tensor K = k_proj.forward(input);
  Tensor V = v_proj.forward(input);

  // 2. Reshape & Transpose for Multi-Head processing
  // Conceptually changing: [batch, seq_len, embed_dim]
  // To: [batch, num_heads, seq_len, head_dim]
  // (Using your zero-copy view manipulators here)
  Q = Q.reshape({/* batch */ 1, /* seq_len */ input.get_shape()[0], num_heads,
                 head_dim})
          .transpose(1, 2);
  K = K.reshape({1, input.get_shape()[0], num_heads, head_dim}).transpose(1, 2);
  V = V.reshape({1, input.get_shape()[0], num_heads, head_dim}).transpose(1, 2);

  // 3. Apply RoPE in-place to Q and K
  rope.apply_in_place(Q, position_offset);
  rope.apply_in_place(K, position_offset);

  // 4. Compute Attention Scores: (Q * K^T) / sqrt(d_k)
  // Needs batched matmul from Project 2
  Tensor K_T = K.transpose(2, 3);
  Tensor scores = Q.matmul(K_T);

  // Scale by 1.0 / sqrt(head_dim)
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  // Apply scalar multiplication (assuming you overloaded operator* in Project
  // 1)
  scores = scores * scale;

  // 5. Causal Masking (only needed during prefill when seq_len > 1)
  if (scores.get_shape().back() > 1) {
    // Iterate and set upper triangle to -INFINITY
    // (Implementation details omitted for brevity)
  }

  // 6. Softmax
  Tensor probs = softmax.forward(scores);

  // 7. Multiply by Values: probs * V
  Tensor context = probs.matmul(V);

  // 8. Reassemble Heads and Output Projection
  // Transpose back: [batch, seq_len, num_heads, head_dim]
  // Reshape to: [seq_len, embed_dim] (2D tensor)
  context =
      context.transpose(1, 2).contiguous().reshape({input.get_shape()[0], embed_dim});

  return o_proj.forward(context);
}

Tensor MultiHeadAttention::forward(const Tensor &input) {
  return forward(input, 0); // Default to position 0 for prefill
}

} // namespace nn
} // namespace turbo