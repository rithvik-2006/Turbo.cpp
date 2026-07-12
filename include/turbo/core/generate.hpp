#pragma once
#include "../model/minigpt.hpp"
#include <vector>

#include "../core/tokenizer.hpp"
#include <string>

namespace turbo {

void generate_text(MiniGPT& model, 
                   Tokenizer& tokenizer, 
                   const std::string& prompt, 
                   int max_tokens, 
                   float temperature = 0.7f);

} // namespace turbo
