#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cstdint>

namespace turbo {

class GGUFLoader; // Forward declaration

enum class TokenizerType { SENTENCEPIECE, BPE };

class Tokenizer {
private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, float> vocab_scores_;
    
    // BPE merge ranks (Priority map)
    std::unordered_map<std::string, int> merge_ranks_;
    
    TokenizerType type_ = TokenizerType::SENTENCEPIECE;
    
    // Isolated Special Tokens
    std::unordered_map<std::string, int> special_tokens_;
    
    // GPT-2 Byte Encoder / Decoder
    std::unordered_map<uint8_t, std::string> byte_encoder_;
    std::unordered_map<std::string, uint8_t> byte_decoder_;

    // Special tokens
    int bos_id_;
    int eos_id_;
    int unk_id_;

    void build_byte_encoder();
    std::string codepoint_to_utf8(int cp);

    // Internal BPE runner
    std::vector<int> run_bpe(const std::string& text);

public:
    Tokenizer(const std::vector<std::string>& vocab, 
              const std::vector<float>& scores,
              const std::vector<std::string>& merges,
              const std::vector<int32_t>& token_types,
              const std::string& model_type,
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
