#include "../../include/turbo/model/minigpt.hpp"
#include "../../include/turbo/core/kv_cache.hpp"
#include "../../include/turbo/loader/gguf_loader.hpp"

namespace turbo {

MiniGPT::MiniGPT(size_t vocab_size, size_t hidden_dim, size_t num_layers,
                 size_t num_heads)
    : tok_emb(vocab_size, hidden_dim), norm(hidden_dim),
      lm_head(hidden_dim, vocab_size), num_heads_(num_heads),
      head_dim_(hidden_dim / num_heads) {

  for (size_t i = 0; i < num_layers; ++i) {
    blocks.emplace_back(hidden_dim, num_heads, num_heads, 4 * hidden_dim);
  }
}

MiniGPT::MiniGPT(GGUFLoader &loader)
    : tok_emb(loader.get_tensor("token_embd.weight")),
      norm(loader.get_tensor("output_norm.weight")),
      lm_head(loader.has_tensor("output.weight") ? loader.get_tensor("output.weight") : loader.get_tensor("token_embd.weight")),
      num_heads_(loader.get_num_heads()),
      num_kv_heads_(loader.get_num_kv_heads()),
      head_dim_(loader.get_hidden_dim() / loader.get_num_heads()) {

  size_t num_layers = loader.get_num_layers();
  size_t hidden_dim = loader.get_hidden_dim();
  float theta_base = loader.get_rope_freq_base();

  std::cout << "  -> Loading Embedding..." << std::endl;
  std::cout << "     rope_freq_base: " << theta_base << std::endl;
  std::cout << "     rope_dim_count: " << loader.get_rope_dim_count() << std::endl;
  
  auto emb_shape = tok_emb.get_weight().get_shape();
  std::cout << "     token_embd.weight shape: [";
  for(auto s : emb_shape) std::cout << s << ", ";
  std::cout << "] dtype: " << static_cast<int>(tok_emb.get_weight().dtype()) << std::endl;

  auto lm_shape = lm_head.get_weight().get_shape();
  std::cout << "     lm_head.weight shape: [";
  for(auto s : lm_shape) std::cout << s << ", ";
  std::cout << "] dtype: " << static_cast<int>(lm_head.get_weight().dtype()) << std::endl;

  // Embedding loaded in init list

  std::cout << "  -> Allocating KV Cache (Will be done by caller)..."
            << std::endl;

  for (size_t i = 0; i < num_layers; ++i) {
    std::cout << "  -> Loading Layer " << i << "..." << std::endl;
    blocks.emplace_back(loader, i, hidden_dim, num_heads_, num_kv_heads_, loader.get_context_length(), theta_base);
  }
}

Tensor MiniGPT::forward(const Tensor &token_ids, KVCache *kv_cache) {
  Tensor x = tok_emb.forward(token_ids).unsqueeze(0);
  size_t seq_len = x.get_shape()[1];
  size_t pos_offset = kv_cache ? kv_cache->get_current_seq_len() : 0;

  // Pass the ENTIRE sequence through the transformer layers
  for (size_t i = 0; i < blocks.size(); ++i) {
      x = blocks[i].forward(x, pos_offset, kv_cache, i);
  }
  x = norm.forward(x);

  // THE SILVER BULLET: Slice the tensor to grab ONLY the last token
  // Slice dim 1 (seq_len) from seq_len-1 to seq_len
  Tensor last_token_x = x.slice(1, seq_len - 1, seq_len);

  // Run the massive LM head matrix multiplication ONLY ONCE for the last token
  Tensor final_logits = lm_head.forward(last_token_x.contiguous());

  return final_logits;
}

Tensor MiniGPT::forward(const std::vector<int> &ids, KVCache *kv_cache) {
  // Convert std::vector<int> to an INT32 Tensor with shape [seq_len]
  std::vector<float> ids_float(ids.begin(),
                               ids.end()); // Internal cast to float for storage
  Tensor input_tensor(ids_float, {ids.size()});
  return forward(input_tensor, kv_cache);
}

} // namespace turbo