#pragma once
#include "turbo/loader/gguf_common.hpp"
#include "turbo/loader/mmap_file.hpp"
#include "turbo/tensor/tensor.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace turbo {

// Holds the architectural rules extracted from the file
struct TensorInfo {
  std::string name;
  uint32_t n_dims;
  std::vector<uint64_t> shape;
  ggml_type type;
  uint64_t offset; // Relative to the start of the tensor data block
};

class GGUFLoader {
private:
  MmapFile file_;

  // Core model hyper-parameters extracted from KV pairs
  uint32_t alignment_ = 32;  // Default GGUF alignment
  uint64_t data_offset_ = 0; // Where the actual weight bytes begin

  // Extracted architecture data
  uint32_t num_layers_ = 32; // Default to LLaMA-7B sizes
  uint32_t hidden_dim_ = 4096;
  uint32_t ffn_intermediate_size_ = 0; // 0 = not parsed yet
  uint32_t num_heads_ = 32;
  uint32_t num_kv_heads_ = 32;
  uint32_t context_length_ = 4096;
  float rope_freq_base_ = 10000.0f;
  uint32_t rope_dim_count_ = 0; // 0 means use head_dim
  std::string chat_template_;

  // Extracted architecture data
  std::unordered_map<std::string, TensorInfo> tensors_;
  std::vector<std::string> vocab_tokens_;
  std::vector<float> vocab_scores_;
  std::vector<int32_t> vocab_token_types_;
  std::string tokenizer_model_ = "llama";
  std::vector<std::string> vocab_merges_;
  int32_t bos_token_id_ = 1;
  int32_t eos_token_id_ = 2;
  int32_t unk_token_id_ = -1;

  // Helper methods for safe pointer traversal
  template <typename T> T read_value(const uint8_t *&ptr);
  std::string read_string(const uint8_t *&ptr);
  void skip_metadata_value(const uint8_t *&ptr, uint32_t type);

public:
  explicit GGUFLoader(const std::string &filepath);

  void parse();
  void print_summary() const;

  const std::vector<std::string> &vocab_tokens() const { return vocab_tokens_; }
  const std::vector<float> &vocab_scores() const { return vocab_scores_; }
  const std::vector<int32_t> &vocab_token_types() const { return vocab_token_types_; }
  const std::vector<std::string> &vocab_merges() const { return vocab_merges_; }
  const std::string &tokenizer_model() const { return tokenizer_model_; }
  int32_t bos_token_id() const { return bos_token_id_; }
  int32_t eos_token_id() const { return eos_token_id_; }
  int32_t unk_token_id() const { return unk_token_id_; }
  const std::string &get_chat_template() const { return chat_template_; }

  // Architecture Getters
  uint32_t get_num_layers() const { return num_layers_; }
  uint32_t get_hidden_dim() const { return hidden_dim_; }
  uint32_t get_ffn_intermediate_size() const { return ffn_intermediate_size_; }
  uint32_t get_num_heads() const { return num_heads_; }
  uint32_t get_num_kv_heads() const { return num_kv_heads_; }
  uint32_t get_context_length() const { return context_length_; }

  Tensor get_tensor(const std::string &name);
  bool has_tensor(const std::string &name) const;
  float get_rope_freq_base() const { return rope_freq_base_; }
  uint32_t get_rope_dim_count() const { return rope_dim_count_; }
};

} // namespace turbo
