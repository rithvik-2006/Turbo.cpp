// include/turbo/nn/rope.hpp
#pragma once
#include "../tensor/tensor.hpp"
#include <vector>

namespace turbo {
namespace nn {

class RoPE {
private:
  size_t head_dim;
  size_t max_seq_len;
  float theta_base;

  // 1D arrays simulating 2D caches: [max_seq_len, head_dim / 2]
  std::vector<float> cos_cache;
  std::vector<float> sin_cache;

public:
  // Precomputes the rotation angles for the maximum expected context window
  RoPE(size_t head_dim, size_t max_seq_len = 4096, float theta_base = 10000.0f);

  // Mutates the tensor in-place.
  // position_offset is crucial for the Decode phase (generating token N+1)
  void apply_in_place(Tensor &x, size_t position_offset = 0);
};

} // namespace nn
} // namespace turbo