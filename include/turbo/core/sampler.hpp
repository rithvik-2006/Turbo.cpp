#pragma once
#include "../tensor/tensor.hpp"

namespace turbo {

int greedy_sample(const float* last_token_logits, size_t vocab_size);

int sample_top_p(const float* last_token_logits, size_t vocab_size, float temperature, float top_p);

} // namespace turbo
