#include "../include/turbo/model/minigpt.hpp"
#include "../include/turbo/core/generate.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Initializing MiniGPT model..." << std::endl;
    
    // Tiny model config for quick local testing
    size_t vocab_size = 32000;
    size_t hidden_dim = 256;
    size_t num_layers = 2;
    size_t num_heads = 4;
    
    turbo::MiniGPT model(vocab_size, hidden_dim, num_layers, num_heads);

    // Mock prompt
    std::vector<int> prompt = {1, 154, 1032, 204}; // Arbitrary tokens
    int max_new_tokens = 50;
    int eos_token_id = 2; // Assuming 2 is EOS

    std::cout << "Starting Autoregressive Loop..." << std::endl;
    std::cout << "Prompt Tokens: ";
    for (int t : prompt) {
        std::cout << t << " ";
    }
    std::cout << "\nOutput: " << std::flush;

    // Run the generation loop (Top-P sampling with Temp = 0.7)
    std::vector<int> output = turbo::generate(model, prompt, max_new_tokens, eos_token_id, false, 0.7f, 0.9f);

    std::cout << "Total Sequence Length: " << output.size() << std::endl;
    return 0;
}
