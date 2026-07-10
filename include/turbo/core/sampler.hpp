#pragma once
#include "../tensor/tensor.hpp"

namespace turbo {

int greedy_sample(const Tensor &logits);

int sample_top_p(const Tensor &logits, float temperature = 1.0f, float top_p = 0.9f);

} // namespace turbo
