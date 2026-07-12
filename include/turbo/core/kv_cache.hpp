#pragma once
#include "turbo/tensor/tensor.hpp"
#include <vector>

namespace turbo {

class KVCache {
private:
    int max_seq_len_;
    int num_layers_;
    int num_kv_heads_;
    int head_dim_;
    
    // We store a vector of Tensors for Keys and Values, one for each layer.
    // Shape of each tensor: [max_seq_len, num_kv_heads, head_dim]
    std::vector<Tensor> k_caches_;
    std::vector<Tensor> v_caches_;

    // Tracks how many tokens are currently in the cache
    int current_seq_len_ = 0;

public:
    KVCache(int max_seq_len, int num_layers, int num_kv_heads, int head_dim);

    // Getters for the Attention layer to update and read the cache
    Tensor& get_k_cache(int layer_idx) { return k_caches_[layer_idx]; }
    Tensor& get_v_cache(int layer_idx) { return v_caches_[layer_idx]; }

    int get_current_seq_len() const { return current_seq_len_; }
    int get_max_seq_len() const { return max_seq_len_; }
    void increment_seq_len(int amount) { current_seq_len_ += amount; }
    void clear() { current_seq_len_ = 0; }
};

} // namespace turbo
