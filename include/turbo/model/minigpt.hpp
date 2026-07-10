#pragma once
#include "../nn/embedding.hpp"
#include "../nn/linear.hpp"
#include "../nn/rmsnorm.hpp"
#include "../nn/transformer_block.hpp"
#include "../tensor/tensor.hpp"
#include <memory>
#include <vector>

namespace turbo {

class MiniGPT {
private:
  nn::Embedding tok_emb;
  std::vector<nn::TransformerBlock> blocks;
  nn::RMSNorm norm;
  nn::Linear lm_head;

public:
  MiniGPT(size_t vocab_size, size_t hidden_dim, size_t num_layers,
          size_t num_heads);

  // The core execution pipeline
  Tensor forward(const Tensor &token_ids);
};

} // namespace turbo