#include "../../include/turbo/model/minigpt.hpp"

namespace turbo {

MiniGPT::MiniGPT(size_t vocab_size, size_t hidden_dim, size_t num_layers,
                 size_t num_heads)
    : tok_emb(vocab_size, hidden_dim), norm(hidden_dim),
      lm_head(hidden_dim, vocab_size) {

  for (size_t i = 0; i < num_layers; ++i) {
    blocks.emplace_back(hidden_dim, num_heads, 4 * hidden_dim);
  }
}

Tensor MiniGPT::forward(const Tensor &token_ids) {
  // 1. Convert integer Token IDs into floating-point vectors
  Tensor x = tok_emb.forward(token_ids);

  // 2. Pass through N Transformer Blocks
  for (auto &block : blocks) {
    x = block.forward(x);
  }

  // 3. Final Pre-computation Normalization
  x = norm.forward(x);

  // 4. Project hidden states back to vocabulary logits
  Tensor logits = lm_head.forward(x);

  return logits;
}

} // namespace turbo