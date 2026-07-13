#include "turbo/core/tokenizer.hpp"
#include "turbo/loader/gguf_loader.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <climits>
#include <algorithm>

namespace turbo {

Tokenizer::Tokenizer(const std::vector<std::string>& vocab, 
                     const std::vector<float>& scores,
                     const std::vector<std::string>& merges,
                     const std::vector<int32_t>& token_types,
                     const std::string& model_type,
                     int bos_id, int eos_id, int unk_id) 
    : vocab_(vocab), bos_id_(bos_id), eos_id_(eos_id), unk_id_(unk_id) {
    
    if (model_type == "gpt2") {
        type_ = TokenizerType::BPE;
        for (size_t i = 0; i < merges.size(); ++i) {
            merge_ranks_[merges[i]] = static_cast<int>(i);
        }
    } else {
        type_ = TokenizerType::SENTENCEPIECE;
    }

    // Build fast lookup maps and merge ranks
    for (size_t i = 0; i < vocab_.size(); ++i) {
        token_to_id_[vocab_[i]] = i;
        if (type_ == TokenizerType::SENTENCEPIECE) {
            if (i < scores.size()) {
                vocab_scores_[vocab_[i]] = scores[i];
            }
        }
    }
    
    // Populate known special tokens for TinyLlama
    special_tokens_["<s>"] = bos_id_;
    special_tokens_["</s>"] = eos_id_;
    // Dynamically register all tokens marked as CONTROL type (3) in GGUF
    for (size_t i = 0; i < vocab_.size(); ++i) {
        if (i < token_types.size() && token_types[i] == 3) {
            special_tokens_[vocab_[i]] = static_cast<int>(i);
        }
    }

    if (type_ == TokenizerType::BPE) {
        build_byte_encoder();
    }
}

Tokenizer::Tokenizer(const GGUFLoader& loader)
    : Tokenizer(loader.vocab_tokens(), loader.vocab_scores(), loader.vocab_merges(), loader.vocab_token_types(), loader.tokenizer_model(), loader.bos_token_id(), loader.eos_token_id(), loader.unk_token_id()) {}

std::string Tokenizer::codepoint_to_utf8(int cp) {
    std::string result;
    if (cp <= 0x7F) {
        result += static_cast<char>(cp);
    } else if (cp <= 0x7FF) {
        result += static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        result += static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0x10FFFF) {
        result += static_cast<char>(0xF0 | ((cp >> 18) & 0x07));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return result;
}

void Tokenizer::build_byte_encoder() {
    std::vector<int> bs;
    for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        std::string utf8_str = codepoint_to_utf8(cs[i]);
        byte_encoder_[static_cast<uint8_t>(bs[i])] = utf8_str;
        byte_decoder_[utf8_str] = static_cast<uint8_t>(bs[i]);
    }
}

std::vector<int> Tokenizer::run_bpe(const std::string& text) {
    if (text.empty()) return {};

    std::string processed_text = text;
    if (type_ == TokenizerType::BPE) {
        std::string remapped;
        for (unsigned char c : text) {
            remapped += byte_encoder_[c];
        }
        processed_text = remapped;
    } else {
        processed_text = "";
        for (char c : text) {
            if (c == ' ') processed_text += "\xe2\x96\x81";
            else processed_text += c;
        }
    }

    // Split text into individual characters/bytes to start
    std::vector<std::string> symbols;
    for (size_t i = 0; i < processed_text.length(); ) {
        unsigned char c = processed_text[i];
        size_t char_len = 1;
        if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;
        
        symbols.push_back(processed_text.substr(i, char_len));
        i += char_len;
    }

    // Iteratively merge
    while (symbols.size() >= 2) {
        int best_idx = -1;
        int lowest_rank = INT_MAX;
        float highest_score = -1e9f;
        std::string best_merge_str = "";

        // Find the adjacent pair with the highest priority score
        for (size_t i = 0; i < symbols.size() - 1; i++) {
            if (type_ == TokenizerType::BPE) {
                std::string pair_str = symbols[i] + " " + symbols[i+1];
                auto it = merge_ranks_.find(pair_str);
                if (it != merge_ranks_.end() && it->second < lowest_rank) {
                    lowest_rank = it->second;
                    best_idx = i;
                    best_merge_str = symbols[i] + symbols[i+1];
                }
            } else {
                std::string pair_str = symbols[i] + symbols[i+1];
                auto it = vocab_scores_.find(pair_str);
                if (it != vocab_scores_.end() && it->second > highest_score) {
                    highest_score = it->second;
                    best_idx = i;
                    best_merge_str = pair_str;
                }
            }
        }

        // If no valid merges are left, we are done
        if (best_idx == -1) {
            break;
        }

        // Merge the best pair
        symbols[best_idx] = best_merge_str;
        symbols.erase(symbols.begin() + best_idx + 1);
    }

    // Convert final string symbols to integer IDs
    std::vector<int> chunk_ids;
    for (const std::string& sym : symbols) {
        auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) {
            chunk_ids.push_back(it->second);
        } else {
            if (type_ == TokenizerType::BPE) {
                std::cerr << "[WARNING] Unmapped BPE symbol: " << sym << "\n";
                if (unk_id_ != -1) chunk_ids.push_back(unk_id_);
            } else {
                // Byte fallback for unknown symbols in SentencePiece
                if (sym.length() == 1) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned char>(sym[0]));
                    auto fallback_it = token_to_id_.find(buf);
                    if (fallback_it != token_to_id_.end()) {
                        chunk_ids.push_back(fallback_it->second);
                    } else if (unk_id_ != -1) {
                        chunk_ids.push_back(unk_id_);
                    }
                } else if (unk_id_ != -1) {
                    chunk_ids.push_back(unk_id_);
                }
            }
        }
    }
    
    return chunk_ids;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) {
    std::vector<int> final_ids;
    if (add_bos && bos_id_ != -1) {
        final_ids.push_back(bos_id_);
    }
    
    std::string current_chunk = "";
    size_t i = 0;
    
    while (i < text.length()) {
        bool found_special = false;
        
        // 1. Check if the current position starts a known special token
        for (const auto& pair : special_tokens_) {
            const std::string& special_str = pair.first;
            int special_id = pair.second;
            
            if (text.substr(i, special_str.length()) == special_str) {
                // Process the accumulated standard text first via BPE
                if (!current_chunk.empty()) {
                    std::vector<int> bpe_ids = run_bpe(current_chunk);
                    final_ids.insert(final_ids.end(), bpe_ids.begin(), bpe_ids.end());
                    current_chunk = "";
                }
                
                // Add the special token ID directly
                final_ids.push_back(special_id);
                i += special_str.length();
                found_special = true;
                break;
            }
        }
        
        // 2. If no special token, accumulate standard text
        if (!found_special) {
            current_chunk += text[i];
            i++;
        }
    }
    
    // Process any remaining standard text
    if (!current_chunk.empty()) {
        std::vector<int> bpe_ids = run_bpe(current_chunk);
        final_ids.insert(final_ids.end(), bpe_ids.begin(), bpe_ids.end());
    }

    return final_ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string text = "";
    for (int id : ids) {
        // Skip special control tokens during output generation
        if (id == bos_id_ || id == eos_id_) continue;
        
        bool is_special = false;
        for (const auto& pair : special_tokens_) {
            if (pair.second == id && id != bos_id_ && id != eos_id_) {
                text += pair.first;
                is_special = true;
                break;
            }
        }
        if (is_special) continue;
        
        if (id >= 0 && static_cast<size_t>(id) < vocab_.size()) {
            text += vocab_[id];
        }
    }
    
    if (type_ == TokenizerType::BPE) {
        std::string decoded_text = "";
        for (size_t i = 0; i < text.length(); ) {
            unsigned char c = text[i];
            size_t char_len = 1;
            if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;
            
            std::string utf8_char = text.substr(i, char_len);
            auto it = byte_decoder_.find(utf8_char);
            if (it != byte_decoder_.end()) {
                decoded_text += static_cast<char>(it->second);
            } else {
                decoded_text += utf8_char;
            }
            i += char_len;
        }
        text = decoded_text;
    } else {
        size_t pos = 0;
        while ((pos = text.find("\xe2\x96\x81", pos)) != std::string::npos) {
            text.replace(pos, 3, " ");
            pos += 1;
        }
    }

    return text;
}

} // namespace turbo
