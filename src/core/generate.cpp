#include "../../include/turbo/core/generate.hpp"
#include "../../include/turbo/core/sampler.hpp"
#include <iostream>

namespace turbo {

std::vector<int> generate(MiniGPT& model, 
                          std::vector<int> prompt_tokens, 
                          int max_new_tokens, 
                          int eos_token_id,
                          bool use_greedy,
                          float temperature, 
                          float top_p) {
    
    std::vector<int> current_sequence = prompt_tokens;

    for (int step = 0; step < max_new_tokens; ++step) {
        std::vector<float> float_sequence(current_sequence.begin(), current_sequence.end());
        std::vector<size_t> shape = {current_sequence.size()};
        Tensor input_tensor(float_sequence, shape); 

        Tensor logits = model.forward(input_tensor);

        int next_token_id;
        if (use_greedy) {
            next_token_id = greedy_sample(logits);
        } else {
            next_token_id = sample_top_p(logits, temperature, top_p);
        }

        current_sequence.push_back(next_token_id);

        std::cout << next_token_id << " " << std::flush;

        if (next_token_id == eos_token_id) {
            break;
        }
    }

    std::cout << std::endl;
    return current_sequence;
}

} // namespace turbo
