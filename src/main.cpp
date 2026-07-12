#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "turbo/loader/gguf_loader.hpp"
#include "turbo/core/tokenizer.hpp"
#include "turbo/model/minigpt.hpp"
#include "turbo/core/sampler.hpp"
#include "turbo/core/kv_cache.hpp"

using namespace turbo;

int main(int argc, char** argv) {
    std::cout << "==========================================\n";
    std::cout << "       TURBO.CPP INFERENCE ENGINE         \n";
    std::cout << "==========================================\n";

    if (argc < 2) {
        std::cerr << "Usage: ./turbo_chat <path_to_model.gguf> [temperature]\n";
        std::cerr << "Example: ./turbo_chat models/llama-2-7b.q4_0.gguf 0.7\n";
        return 1;
    }

    std::string model_path = argv[1];
    float temp = (argc >= 3) ? std::stof(argv[2]) : 0.8f;

    try {
        auto start_load = std::chrono::high_resolution_clock::now();

        // 1. Map the weights directly from the SSD (Zero-Copy)
        std::cout << "[INFO] Memory mapping model: " << model_path << "...\n";
        GGUFLoader loader(model_path);
        loader.parse(); // Added this explicitly since GGUFLoader parsing might not be entirely in the constructor.

        // 2. Initialize the Tokenizer from GGUF metadata
        std::cout << "[INFO] Loading BPE vocabulary...\n";
        Tokenizer tokenizer(loader);

        // 3. Boot up the Neural Network
        std::cout << "[INFO] Initializing Transformer architecture...\n";
        MiniGPT model(loader);
        
        // 4. Initialize KV Cache for multi-turn context
        // Use dynamically loaded architecture sizes
        // Reduce seq_len to 256 to avoid FP32 memory explosion
        KVCache kv_cache(256, 
                         model.get_num_layers(), 
                         model.get_num_kv_heads(), 
                         model.get_head_dim());

        auto end_load = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> load_time = end_load - start_load;
        std::cout << "[INFO] Engine online. Boot time: " << load_time.count() << " seconds.\n\n";

        // --- THE INACTIVE CHAT LOOP ---
        std::cout << "Entering Chat Mode (type 'exit' or 'quit' to end session)\n";
        std::cout << "==========================================\n";
        
        std::string prompt;
        while (true) {
            std::cout << "\nUser: ";
            if (!std::getline(std::cin, prompt)) {
                break;
            }
            if (prompt == "exit" || prompt == "quit") {
                break;
            }
            if (prompt.empty()) {
                continue;
            }

            std::cout << "Turbo: ";

            // Format the prompt for TinyLlama Chat architecture
            std::string formatted_prompt = "<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n" + prompt + "</s>\n<|assistant|>\n";
            
            // 5. Encode the human text into an array of integer IDs
            std::vector<int> input_ids = tokenizer.encode(formatted_prompt, true);
            
            // Reset the KV cache for each new conversation turn
            kv_cache.clear();

            int max_tokens = 512;
            int token_count = 0;
            
            auto start_gen = std::chrono::high_resolution_clock::now();

            // 6. The Autoregressive Generation Loop
            for (int i = 0; i < max_tokens; i++) {
                // Guard: Prevent KV cache overflow before the forward pass
                int needed = kv_cache.get_current_seq_len() + static_cast<int>(input_ids.size());
                if (needed > kv_cache.get_max_seq_len()) {
                    std::cout << "\n[INFO] KV cache full (" << needed << "/" << kv_cache.get_max_seq_len() << "). Stopping generation.";
                    break;
                }

                // Forward pass: Compute attention and FFN layers
                Tensor logits = model.forward(input_ids, &kv_cache);
                kv_cache.increment_seq_len(input_ids.size());

                if (i == 0) {
                    std::cout << "\n[DEBUG] Logits shape: [";
                    for (auto s : logits.get_shape()) std::cout << s << ", ";
                    std::cout << "]\n";
                    
                    // Find argmax of last token logits
                    size_t vocab_size = 32000;
                    const float* last_logits = logits.data_ptr() + logits.numel() - vocab_size;
                    
                    // Check for NaNs and INFs
                    int nan_count = 0;
                    int inf_count = 0;
                    float max_val = -INFINITY;
                    int argmax_id = 0;
                    
                    for (size_t j = 0; j < vocab_size; ++j) {
                        float v = last_logits[j];
                        if (std::isnan(v)) nan_count++;
                        else if (std::isinf(v)) inf_count++;
                        
                        if (j > 0 && v > max_val) {
                            max_val = v;
                            argmax_id = static_cast<int>(j);
                        } else if (j == 0) {
                            max_val = v;
                        }
                    }
                    std::cout << "[DEBUG] NaNs: " << nan_count << ", INFs: " << inf_count << "\n";
                    std::cout << "[DEBUG] Greedy (argmax) token: " << argmax_id 
                              << " ('" << tokenizer.decode({argmax_id}) << "') logit=" << max_val << "\n";
                    std::cout << "[DEBUG] First 10 logits (last token): ";
                    for (int j = 0; j < 10; ++j) std::cout << last_logits[j] << " ";
                    std::cout << "\n";
                }

                // Sample the next token ID based on our temperature/Top-P
                int next_token;
                if (temp <= 0.0f) {
                    next_token = greedy_sample(logits);
                } else {
                    next_token = sample_top_p(logits, temp, 0.9f);
                }

                // Break the loop if the model naturally decides it is done talking
                if (next_token == tokenizer.eos_id()) {
                    break;
                }

                // Decode the raw integer back into a human string and print it instantly
                std::string token_str = tokenizer.decode({next_token});
                std::cout << token_str << std::flush;
                token_count++;

                // Feed the generated token back in as the input for the next cycle
                input_ids = {next_token};
            }

            auto end_gen = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> gen_time = end_gen - start_gen;
            
            std::cout << "\n\n[INFO] Generated " << token_count << " tokens. Speed: " << (token_count / gen_time.count()) << " tokens/second.\n";
            std::cout << "------------------------------------------\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
