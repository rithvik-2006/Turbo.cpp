#include "../include/turbo/model/minigpt.hpp"
#include "../include/turbo/core/generate.hpp"
#include "../include/turbo/core/tokenizer.hpp"
#include "../include/turbo/loader/gguf_loader.hpp"
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

    std::cout << "Loading Tokenizer from dummy.gguf..." << std::endl;
    turbo::GGUFLoader loader("dummy.gguf");
    loader.parse();
    
    turbo::Tokenizer tokenizer(loader.vocab_tokens(), loader.vocab_scores());

    // Mock prompt
    std::string prompt = "Hello World, this is the Turbo Engine speaking.";
    int max_new_tokens = 50;

    std::cout << "Starting Autoregressive Loop..." << std::endl;
    
    // Run the generation loop streaming to console
    turbo::generate_text(model, tokenizer, prompt, max_new_tokens, 0.7f);

    return 0;
}
