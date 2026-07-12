#include "../../include/turbo/model/minigpt.hpp"
#include "../../include/turbo/core/kv_cache.hpp"
#include "../../include/turbo/loader/gguf_loader.hpp"

namespace turbo {

MiniGPT::MiniGPT(size_t vocab_size, size_t hidden_dim, size_t num_layers,
                 size_t num_heads)
    : tok_emb(vocab_size, hidden_dim), norm(hidden_dim),
      lm_head(hidden_dim, vocab_size), num_heads_(num_heads), head_dim_(hidden_dim / num_heads) {

  for (size_t i = 0; i < num_layers; ++i) {
    blocks.emplace_back(hidden_dim, num_heads, num_heads, 4 * hidden_dim);
  }
}

MiniGPT::MiniGPT(GGUFLoader& loader) 
    : tok_emb(loader.get_tensor("token_embd.weight")),
      norm(loader.get_tensor("output_norm.weight")),
      lm_head(loader.get_tensor("output.weight")),
      num_heads_(loader.get_num_heads()), 
      num_kv_heads_(loader.get_num_kv_heads()),
      head_dim_(loader.get_hidden_dim() / loader.get_num_heads()) {

    size_t num_layers = loader.get_num_layers();
    size_t hidden_dim = loader.get_hidden_dim();

    std::cout << "  -> Loading Embedding..." << std::endl;
    // Embedding loaded in init list

    std::cout << "  -> Allocating KV Cache (Will be done by caller)..." << std::endl;

    for (size_t i = 0; i < num_layers; ++i) {
        std::cout << "  -> Loading Layer " << i << "..." << std::endl;
        blocks.emplace_back(loader, i, hidden_dim, num_heads_, num_kv_heads_);
    }
}

Tensor MiniGPT::forward(const Tensor &token_ids, KVCache* kv_cache) {
  // 1. Convert integer Token IDs into floating-point vectors and restore the batch dimension [1, seq_len, embed_dim]
  Tensor x = tok_emb.forward(token_ids).unsqueeze(0);
  
  // Extract tracking info for RoPE positional embeddings
  size_t pos_offset = kv_cache ? kv_cache->get_current_seq_len() : 0;

  // 2. Pass through N Transformer Blocks
  for (size_t i = 0; i < blocks.size(); ++i) {
    x = blocks[i].forward(x, pos_offset, kv_cache, i);
  }

  // 3. Final Pre-computation Normalization
  x = norm.forward(x);

  // 4. Project hidden states back to vocabulary logits
  Tensor logits = lm_head.forward(x);

  return logits;
}

Tensor MiniGPT::forward(const std::vector<int>& ids, KVCache* kv_cache) {
    // Convert std::vector<int> to an INT32 Tensor with shape [seq_len]
    std::vector<float> ids_float(ids.begin(), ids.end()); // Internal cast to float for storage
    Tensor input_tensor(ids_float, {ids.size()});
    return forward(input_tensor, kv_cache);
}

} // namespace turbo