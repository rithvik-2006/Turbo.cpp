#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>

namespace turbo {

class GGUFLoader; // Forward declaration

class Tokenizer {
private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, float> vocab_scores_;
    
    // BPE merge ranks (Priority map)
    std::unordered_map<std::string, int> merge_ranks_;
    
    // Isolated Special Tokens
    std::unordered_map<std::string, int> special_tokens_;
    
    // Special tokens
    int bos_id_;
    int eos_id_;
    int unk_id_;

    // Internal BPE runner
    std::vector<int> run_bpe(const std::string& text);

public:
    Tokenizer(const std::vector<std::string>& vocab, 
              const std::vector<float>& scores,
              int bos_id = 1, int eos_id = 2, int unk_id = 0);

    // Convenience constructor for GGUFLoader
    explicit Tokenizer(const GGUFLoader& loader);

    int eos_id() const { return eos_id_; }
    int bos_id() const { return bos_id_; }

    // Core APIs
    std::vector<int> encode(const std::string& text, bool add_bos = true);
    std::string decode(const std::vector<int>& ids);
};

} // namespace turbo
