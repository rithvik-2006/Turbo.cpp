#pragma once
#include "../nn/embedding.hpp"
#include "../nn/linear.hpp"
#include "../nn/rmsnorm.hpp"
#include "../nn/transformer_block.hpp"
#include "../tensor/tensor.hpp"
#include <memory>
#include <vector>

namespace turbo {

class KVCache;

class GGUFLoader; // Forward declaration

class MiniGPT {
private:
  nn::Embedding tok_emb;
  std::vector<nn::TransformerBlock> blocks;
  nn::RMSNorm norm;
  nn::Linear lm_head;
  
  size_t num_heads_;
  size_t num_kv_heads_;
  size_t head_dim_;

public:
  MiniGPT(size_t vocab_size, size_t hidden_dim, size_t num_layers,
          size_t num_heads);
          
  explicit MiniGPT(GGUFLoader& loader);

  // Getters for KVCache initialization
  size_t get_num_layers() const { return blocks.size(); }
  size_t get_num_heads() const { return num_heads_; }
  size_t get_num_kv_heads() const { return num_kv_heads_; }
  size_t get_head_dim() const { return head_dim_; }

  // The core execution pipeline
  Tensor forward(const Tensor &token_ids, KVCache* kv_cache = nullptr);
  Tensor forward(const std::vector<int>& ids, KVCache* kv_cache = nullptr);
};

} // namespace turbo