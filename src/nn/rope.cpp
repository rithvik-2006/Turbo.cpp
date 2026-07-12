// src/nn/rope.cpp
#include "../../include/turbo/nn/rope.hpp"
#include <cmath>
#include <stdexcept>

namespace turbo {
namespace nn {

RoPE::RoPE(size_t head_dim, size_t max_seq_len, float theta_base)
    : head_dim(head_dim), max_seq_len(max_seq_len), theta_base(theta_base) {

  if (head_dim % 2 != 0) {
    throw std::invalid_argument("RoPE requires an even head dimension.");
  }

  size_t half_dim = head_dim / 2;
  cos_cache.resize(max_seq_len * half_dim);
  sin_cache.resize(max_seq_len * half_dim);

  // Precompute the rotation frequencies
  for (size_t pos = 0; pos < max_seq_len; ++pos) {
    for (size_t i = 0; i < half_dim; ++i) {
      // Calculate theta_i = 10000^(-2i / d)
      float theta_i =
          std::pow(theta_base, -static_cast<float>(2 * i) / head_dim);
      float freq = pos * theta_i;

      size_t cache_idx = pos * half_dim + i;
      cos_cache[cache_idx] = std::cos(freq);
      sin_cache[cache_idx] = std::sin(freq);
    }
  }
}

void RoPE::apply_in_place(Tensor &x, size_t position_offset) {
  auto shape = x.get_shape();
  if (shape.empty() || shape.back() != head_dim) {
    throw std::invalid_argument(
        "Tensor last dimension must match RoPE head_dim.");
  }

  // Assume shape is [..., seq_len, head_dim]
  // For a robust engine, you'd calculate total outer loops, but for clarity
  // we assume a 2D slice: [seq_len, head_dim]
  size_t seq_len = shape[shape.size() - 2];
  size_t half_dim = head_dim / 2;

  if (shape.size() == 4) {
    size_t batch = shape[0];
    
    // After transpose in attention.cpp, layout is [batch, num_heads, seq_len, head_dim]
    size_t num_heads_dim = shape[1];
    size_t seq_len_dim = shape[2];
    
    for (size_t b = 0; b < batch; ++b) {
      for (size_t pos = 0; pos < seq_len_dim; ++pos) {
        size_t absolute_pos = position_offset + pos;
        if (absolute_pos >= max_seq_len) {
          throw std::out_of_range("Sequence position exceeds RoPE cache size.");
        }

        for (size_t h = 0; h < num_heads_dim; ++h) {
          for (size_t i = 0; i < half_dim; ++i) {
            size_t cache_idx = absolute_pos * half_dim + i;
            float cos_val = cos_cache[cache_idx];
            float sin_val = sin_cache[cache_idx];

            // GGUF permutes weights for interleaved RoPE
            float x0 = x.at({b, h, pos, i * 2});
            float x1 = x.at({b, h, pos, i * 2 + 1});

            x.at({b, h, pos, i * 2}) = x0 * cos_val - x1 * sin_val;
            x.at({b, h, pos, i * 2 + 1}) = x0 * sin_val + x1 * cos_val;
          }
        }
      }
    }
  } else if (shape.size() == 2) {
    for (size_t pos = 0; pos < seq_len; ++pos) {
      size_t absolute_pos = position_offset + pos;
      if (absolute_pos >= max_seq_len) {
        throw std::out_of_range("Sequence position exceeds RoPE cache size.");
      }

      for (size_t i = 0; i < half_dim; ++i) {
        size_t cache_idx = absolute_pos * half_dim + i;
        float cos_val = cos_cache[cache_idx];
        float sin_val = sin_cache[cache_idx];

        float x0 = x.at({pos, i * 2});
        float x1 = x.at({pos, i * 2 + 1});

        x.at({pos, i * 2}) = x0 * cos_val - x1 * sin_val;
        x.at({pos, i * 2 + 1}) = x0 * sin_val + x1 * cos_val;
      }
    }
  } else {
    throw std::invalid_argument("RoPE apply_in_place currently supports 2D or 4D tensors.");
  }
}

} // namespace nn
} // namespace turbo