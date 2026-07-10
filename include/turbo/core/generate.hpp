#pragma once
#include "../model/minigpt.hpp"
#include <vector>

namespace turbo {

std::vector<int> generate(MiniGPT& model, 
                          std::vector<int> prompt_tokens, 
                          int max_new_tokens, 
                          int eos_token_id,
                          bool use_greedy = false,
                          float temperature = 1.0f, 
                          float top_p = 0.9f);

} // namespace turbo
