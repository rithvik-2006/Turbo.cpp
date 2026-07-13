#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

#include "turbo/core/kv_cache.hpp"
#include "turbo/core/sampler.hpp"
#include "turbo/core/tokenizer.hpp"
#include "turbo/loader/gguf_loader.hpp"
#include "turbo/model/minigpt.hpp"

using namespace turbo;

int main(int argc, char **argv) {
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

  if (!std::filesystem::exists(model_path)) {
    if (std::filesystem::exists("build/" + model_path)) {
      model_path = "build/" + model_path;
    } else if (std::filesystem::exists("models/" + model_path)) {
      model_path = "models/" + model_path;
    }
  }

  try {
    auto start_load = std::chrono::high_resolution_clock::now();

    // 1. Map the weights directly from the SSD (Zero-Copy)
    std::cout << "[INFO] Memory mapping model: " << model_path << "...\n";
    GGUFLoader loader(model_path);
    loader.parse(); // Added this explicitly since GGUFLoader parsing might not
                    // be entirely in the constructor.

    // 2. Initialize the Tokenizer from GGUF metadata
    std::cout << "[INFO] Loading BPE vocabulary...\n";
    Tokenizer tokenizer(loader);

    // 3. Boot up the Neural Network
    std::cout << "[INFO] Initializing Transformer architecture...\n";
    
    // Find tokens for Hello and llama
    const auto& vocab = loader.vocab_tokens();
    std::cout << "Searching vocab for tokens...\n";
    for (size_t i = 0; i < vocab.size(); ++i) {
        if (vocab[i] == "<|begin_of_text|>" || vocab[i] == "Hello" || vocab[i] == "Ġllama" || vocab[i] == "llama") {
            std::cout << "Token ID for '" << vocab[i] << "': " << i << "\n";
        }
    }
    
    MiniGPT model(loader);

    // 4. Initialize KV Cache for multi-turn context
    // Use dynamically loaded architecture sizes
    // Reduce seq_len to 256 to avoid FP32 memory explosion
    KVCache kv_cache(256, model.get_num_layers(), model.get_num_kv_heads(),
                     model.get_head_dim());

    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_time = end_load - start_load;
    std::cout << "[INFO] Engine online. Boot time: " << load_time.count()
              << " seconds.\n\n";

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

      // Format the prompt for Llama-3 Chat architecture
      std::string formatted_prompt =
          "<|start_header_id|>system<|end_header_id|>\n\n"
          "You are a helpful AI assistant.<|eot_id|>"
          "<|start_header_id|>user<|end_header_id|>\n\n" + 
          prompt + "<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n";

      // 5. Encode the human text into an array of integer IDs
      std::vector<int> input_ids = tokenizer.encode(formatted_prompt, true);
      // std::vector<int> input_ids = {128000, 9906, 94776};

      // Reset the KV cache for each new conversation turn
      kv_cache.clear();

      int max_tokens = 512;
      int token_count = 0;

      auto start_gen = std::chrono::high_resolution_clock::now();

      // 6. The Autoregressive Generation Loop
      for (int i = 0; i < max_tokens; i++) {
        // Guard: Prevent KV cache overflow before the forward pass
        int needed =
            kv_cache.get_current_seq_len() + static_cast<int>(input_ids.size());
        if (needed > kv_cache.get_max_seq_len()) {
          std::cout << "\n[INFO] KV cache full (" << needed << "/"
                    << kv_cache.get_max_seq_len() << "). Stopping generation.";
          break;
        }

        // Forward pass: Compute attention and FFN layers
        Tensor logits = model.forward(input_ids, &kv_cache);
        
        // We now increment KV cache manually here for the entire sequence passed
        kv_cache.increment_seq_len(input_ids.size());



        // Slicer for Fix 2: Advance the pointer to the very last token's
        // predictions
        int seq_len_current = logits.get_shape()[1];
        int vocab_size_current = logits.get_shape()[2];
        float *last_token_logits =
            logits.data_ptr() + ((seq_len_current - 1) * vocab_size_current);

        // Sample the next token ID based on our temperature/Top-P
        int next_token;
        if (temp <= 0.0f) {
          next_token = greedy_sample(last_token_logits, vocab_size_current);
        } else {
          next_token =
              sample_top_p(last_token_logits, vocab_size_current, temp, 0.9f);
        }

        // Break the loop if the model naturally decides it is done talking
        if (next_token == tokenizer.eos_id() || next_token == 128009) {
          std::cout << "\n[Turn Complete]" << std::endl;
          break;
        }

        // Decode the raw integer back into a human string and print it
        // instantly
        // std::cout << "[" << next_token << "] " << std::flush;
        std::string token_str = tokenizer.decode({next_token});
        std::cout << token_str << std::flush;
        token_count++;

        // Feed the generated token back in as the input for the next cycle
        input_ids = {next_token};
      }

      auto end_gen = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> gen_time = end_gen - start_gen;

      std::cout << "\n\n[INFO] Generated " << token_count
                << " tokens. Speed: " << (token_count / gen_time.count())
                << " tokens/second.\n";
      std::cout << "------------------------------------------\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "\n[FATAL ERROR] " << e.what() << "\n";
    return 1;
  }

  return 0;
}
