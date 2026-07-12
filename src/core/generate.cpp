#include "../../include/turbo/core/generate.hpp"
#include "../../include/turbo/core/sampler.hpp"
#include "../../include/turbo/core/kv_cache.hpp"
#include "../../include/turbo/core/tokenizer.hpp"
#include <iostream>

namespace turbo {

void generate_text(MiniGPT& model, Tokenizer& tokenizer, const std::string& prompt, int max_tokens, float temperature) {
    // 1. Initialize State
    std::vector<int> prompt_ids = tokenizer.encode(prompt);
    
    // Convert prompt to a float vector for our tensor
    std::vector<float> prompt_float(prompt_ids.begin(), prompt_ids.end());
    
    // Shape: [seq_len] (1D tensor for embeddings)
    Tensor input_tokens(prompt_float, {prompt_ids.size()}); 
    
    // Initialize KV Cache
    int max_seq_len = 2048;
    KVCache kv_cache(max_seq_len, 
                     model.get_num_layers(), 
                     model.get_num_kv_heads(), 
                     model.get_head_dim());

    std::cout << prompt << std::flush;

    // 2. The Prefill Phase
    // Pass the entire prompt through the model to compute initial logits and fill the cache
    Tensor logits = model.forward(input_tokens, &kv_cache);
    
    // Update the cache length by the number of tokens in the prompt
    kv_cache.increment_seq_len(prompt_ids.size());

    // 3. The Decode Phase
    int generated_count = 0;
    while (generated_count < max_tokens) {
        // Sample the next token ID from the final position's logits
        int next_id;
        if (temperature > 0.0f) {
            next_id = sample_top_p(logits, temperature, 0.9f);
        } else {
            next_id = greedy_sample(logits);
        }
        
        // Break if we hit the End-of-Sequence token
        if (next_id == tokenizer.eos_id()) {
            break;
        }

        // Decode and print in real-time
        std::cout << tokenizer.decode({next_id}) << std::flush;

        // Prepare the single new token as the next input: shape [1]
        std::vector<float> single_token_float = {static_cast<float>(next_id)};
        input_tokens = Tensor(single_token_float, {1});

        // Forward pass: Compute attention using only the new token + KV Cache history
        logits = model.forward(input_tokens, &kv_cache);
        
        // Increment the cache sequence length by 1 for the newly processed token
        kv_cache.increment_seq_len(1);
        generated_count++;
    }
    std::cout << std::endl;
}

} // namespace turbo
