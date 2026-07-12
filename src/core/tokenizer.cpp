#include "turbo/core/tokenizer.hpp"
#include "turbo/loader/gguf_loader.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <climits>

namespace turbo {

Tokenizer::Tokenizer(const std::vector<std::string>& vocab, 
                     const std::vector<float>& scores,
                     int bos_id, int eos_id, int unk_id) 
    : vocab_(vocab), bos_id_(bos_id), eos_id_(eos_id), unk_id_(unk_id) {
    
    // Build fast lookup maps and merge ranks
    for (size_t i = 0; i < vocab_.size(); ++i) {
        token_to_id_[vocab_[i]] = i;
        if (i < scores.size()) {
            vocab_scores_[vocab_[i]] = scores[i];
            merge_ranks_[vocab_[i]] = static_cast<int>(scores.size() - i); // Use inverse index as rank if scores aren't direct ranks
        } else {
            token_to_id_[vocab_[i]] = i;
        }
    }
    
    // Populate known special tokens for TinyLlama
    special_tokens_["<s>"] = bos_id_;
    special_tokens_["</s>"] = eos_id_;
    
    // Try to find chat template special tokens in vocab to map them
    std::vector<std::string> chat_tokens = {"<|system|>", "<|user|>", "<|assistant|>"};
    for (const auto& ct : chat_tokens) {
        auto it = token_to_id_.find(ct);
        if (it != token_to_id_.end()) {
            special_tokens_[ct] = it->second;
        }
    }
}

Tokenizer::Tokenizer(const GGUFLoader& loader)
    : Tokenizer(loader.vocab_tokens(), loader.vocab_scores(), 1, 2, 0) {}

std::vector<int> Tokenizer::run_bpe(const std::string& text) {
    if (text.empty()) return {};

    // TinyLlama usually requires prepending the Meta Space (U+2581) to words
    std::string processed_text = "";
    for (char c : text) {
        if (c == ' ') processed_text += "\xe2\x96\x81";
        else processed_text += c;
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
        float highest_score = -1e9f;
        std::string best_merge_str = "";

        // Find the adjacent pair with the highest priority score
        for (size_t i = 0; i < symbols.size() - 1; i++) {
            std::string pair_str = symbols[i] + symbols[i+1];
            
            auto it = vocab_scores_.find(pair_str);
            if (it != vocab_scores_.end() && it->second > highest_score) {
                highest_score = it->second;
                best_idx = i;
                best_merge_str = pair_str;
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
            // Byte fallback for unknown symbols
            if (sym.length() == 1) {
                char buf[7];
                snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned char>(sym[0]));
                auto fallback_it = token_to_id_.find(buf);
                if (fallback_it != token_to_id_.end()) {
                    chunk_ids.push_back(fallback_it->second);
                } else {
                    chunk_ids.push_back(unk_id_);
                }
            } else {
                chunk_ids.push_back(unk_id_);
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

    std::cout << "[TOKENIZER DEBUG] Final tokens: ";
    for (int t : final_ids) {
        if (t >= 0 && static_cast<size_t>(t) < vocab_.size()) {
            std::cout << "[" << t << ": '" << vocab_[t] << "'] ";
        } else {
            std::cout << "[" << t << ": 'SPECIAL'] ";
        }
    }
    std::cout << "\n";
    
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
    
    // SentencePiece De-normalization: replace U+2581 back with ' '
    size_t pos = 0;
    while ((pos = text.find("\xe2\x96\x81", pos)) != std::string::npos) {
        text.replace(pos, 3, " ");
        pos += 1;
    }

    return text;
}

} // namespace turbo
