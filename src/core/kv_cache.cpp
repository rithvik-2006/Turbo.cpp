#include "turbo/core/kv_cache.hpp"

namespace turbo {

KVCache::KVCache(int max_seq_len, int num_layers, int num_kv_heads, int head_dim)
    : max_seq_len_(max_seq_len), num_layers_(num_layers), 
      num_kv_heads_(num_kv_heads), head_dim_(head_dim) {
    
    for (int i = 0; i < num_layers_; ++i) {
        size_t numel = max_seq_len_ * num_kv_heads_ * head_dim_;
        std::vector<float> zero_data(numel, 0.0f);
        std::vector<size_t> shape = {static_cast<size_t>(max_seq_len_), 
                                     static_cast<size_t>(num_kv_heads_), 
                                     static_cast<size_t>(head_dim_)};
        
        k_caches_.push_back(Tensor(zero_data, shape));
        v_caches_.push_back(Tensor(zero_data, shape));
    }
}

} // namespace turbo
