#include "turbo/loader/gguf_loader.hpp"
#include <cstring>
#include <iostream>

namespace turbo {

// --- Pointer Arithmetic Helpers ---

DataType map_ggml_type(uint32_t ggml_type) {
  switch (ggml_type) {
  case GGML_TYPE_F32:
    return DataType::FP32;
  case GGML_TYPE_F16:
    return DataType::FP16;
  case GGML_TYPE_Q8_0:
    return DataType::Q8_0;
  case GGML_TYPE_Q4_0:
    return DataType::Q4_0;
  default:
    throw std::runtime_error("Unsupported GGUF tensor type encountered: " +
                             std::to_string(ggml_type));
  }
}

template <typename T> T GGUFLoader::read_value(const uint8_t *&ptr) {
  T val;
  std::memcpy(&val, ptr, sizeof(T));
  ptr += sizeof(T);
  return val;
}

std::string GGUFLoader::read_string(const uint8_t *&ptr) {
  uint64_t len = read_value<uint64_t>(ptr);
  std::string s(reinterpret_cast<const char *>(ptr), len);
  ptr += len;
  return s;
}

// Skips over dynamic metadata payloads so we don't crash reading the next key
void GGUFLoader::skip_metadata_value(const uint8_t *&ptr, uint32_t type) {
  switch (type) {
  case GGUF_TYPE_UINT8:
  case GGUF_TYPE_INT8:
    ptr += 1;
    break;
  case GGUF_TYPE_UINT16:
  case GGUF_TYPE_INT16:
    ptr += 2;
    break;
  case GGUF_TYPE_UINT32:
  case GGUF_TYPE_INT32:
  case GGUF_TYPE_FLOAT32:
    ptr += 4;
    break;
  case GGUF_TYPE_UINT64:
  case GGUF_TYPE_INT64:
  case GGUF_TYPE_FLOAT64:
    ptr += 8;
    break;
  case GGUF_TYPE_BOOL:
    ptr += 1;
    break;
  case GGUF_TYPE_STRING:
    read_string(ptr);
    break;
  case GGUF_TYPE_ARRAY: {
    uint32_t arr_type = read_value<uint32_t>(ptr);
    uint64_t arr_len = read_value<uint64_t>(ptr);
    for (uint64_t i = 0; i < arr_len; ++i) {
      skip_metadata_value(ptr, arr_type);
    }
    break;
  }
  default:
    throw std::runtime_error("Unknown GGUF metadata type");
  }
}

// --- Core Loader ---

GGUFLoader::GGUFLoader(const std::string &filepath) : file_(filepath) {}

void GGUFLoader::parse() {
  const uint8_t *ptr = static_cast<const uint8_t *>(file_.data());

  // 1. Read and validate header
  auto header = read_value<gguf_header_t>(ptr);
  if (header.magic != GGUF_MAGIC) {
    throw std::runtime_error("Invalid GGUF magic bytes. Not a GGUF file.");
  }

  std::cout << "GGUF Version: " << header.version << "\n";
  std::cout << "KV Pairs: " << header.kv_count
            << " | Tensors: " << header.tensor_count << "\n";

  // 2. Parse KV Metadata
  for (uint64_t i = 0; i < header.kv_count; ++i) {
    std::string key = read_string(ptr);
    uint32_t type = read_value<uint32_t>(ptr);

    std::cout << "[GGUF] Key: " << key << " (type: " << type << ")\n";

    // Extract alignment if present (critical for calculating final tensor
    // offsets)
    if (key == "general.alignment" && type == GGUF_TYPE_UINT32) {
      alignment_ = read_value<uint32_t>(ptr);
    } else if (key == "llama.block_count" && type == GGUF_TYPE_UINT32) {
      num_layers_ = read_value<uint32_t>(ptr);
    } else if (key == "llama.embedding_length" && type == GGUF_TYPE_UINT32) {
      hidden_dim_ = read_value<uint32_t>(ptr);
    } else if (key == "llama.feed_forward_length" && type == GGUF_TYPE_UINT32) {
      ffn_intermediate_size_ = read_value<uint32_t>(ptr);
    } else if (key == "llama.attention.head_count" &&
               type == GGUF_TYPE_UINT32) {
      num_heads_ = read_value<uint32_t>(ptr);
    } else if (key == "llama.attention.head_count_kv" &&
               type == GGUF_TYPE_UINT32) {
      num_kv_heads_ = read_value<uint32_t>(ptr);
    } else if (key == "llama.context_length" && type == GGUF_TYPE_UINT32) {
      context_length_ = read_value<uint32_t>(ptr);
    } else if (key == "tokenizer.ggml.tokens" && type == GGUF_TYPE_ARRAY) {
      uint32_t arr_type = read_value<uint32_t>(ptr);
      uint64_t arr_len = read_value<uint64_t>(ptr);
      if (arr_type != GGUF_TYPE_STRING) {
        throw std::runtime_error(
            "tokenizer.ggml.tokens must be an array of strings.");
      }
      vocab_tokens_.reserve(arr_len);
      for (uint64_t j = 0; j < arr_len; ++j) {
        vocab_tokens_.push_back(read_string(ptr));
      }
    } else if (key == "tokenizer.ggml.scores" && type == GGUF_TYPE_ARRAY) {
      uint32_t arr_type = read_value<uint32_t>(ptr);
      uint64_t arr_len = read_value<uint64_t>(ptr);
      if (arr_type != GGUF_TYPE_FLOAT32) {
        throw std::runtime_error(
            "tokenizer.ggml.scores must be an array of floats.");
      }
      vocab_scores_.reserve(arr_len);
      for (uint64_t j = 0; j < arr_len; ++j) {
        vocab_scores_.push_back(read_value<float>(ptr));
      }
    } else if (key == "tokenizer.ggml.token_type" && type == GGUF_TYPE_ARRAY) {
      uint32_t arr_type = read_value<uint32_t>(ptr);
      uint64_t arr_len = read_value<uint64_t>(ptr);
      if (arr_type != GGUF_TYPE_INT32) {
        throw std::runtime_error("tokenizer.ggml.token_type must be an array of int32.");
      }
      vocab_token_types_.reserve(arr_len);
      for (uint64_t j = 0; j < arr_len; ++j) {
        vocab_token_types_.push_back(read_value<int32_t>(ptr));
      }
    } else if (key == "tokenizer.ggml.model" && type == GGUF_TYPE_STRING) {
      tokenizer_model_ = read_string(ptr);
    } else if (key == "tokenizer.ggml.merges" && type == GGUF_TYPE_ARRAY) {
      uint32_t arr_type = read_value<uint32_t>(ptr);
      uint64_t arr_len = read_value<uint64_t>(ptr);
      if (arr_type != GGUF_TYPE_STRING) {
        throw std::runtime_error(
            "tokenizer.ggml.merges must be an array of strings.");
      }
      vocab_merges_.reserve(arr_len);
      for (uint64_t j = 0; j < arr_len; ++j) {
        vocab_merges_.push_back(read_string(ptr));
      }
    } else if (key == "llama.attention.value_length" && type == GGUF_TYPE_UINT32) {
      // Sometimes separate KV head dims are specified, we skip for now if head_dim is uniform
      skip_metadata_value(ptr, type);
    } else if (key == "llama.rope.freq_base" && type == GGUF_TYPE_FLOAT32) {
      rope_freq_base_ = read_value<float>(ptr);
    } else if (key == "llama.rope.dimension_count" && type == GGUF_TYPE_UINT32) {
      rope_dim_count_ = read_value<uint32_t>(ptr);
    } else if (key == "tokenizer.ggml.bos_token_id" && type == GGUF_TYPE_UINT32) {
      bos_token_id_ = read_value<uint32_t>(ptr);
    } else if (key == "tokenizer.ggml.eos_token_id" && type == GGUF_TYPE_UINT32) {
      eos_token_id_ = read_value<uint32_t>(ptr);
    } else if (key == "tokenizer.ggml.unknown_token_id" && type == GGUF_TYPE_UINT32) {
      unk_token_id_ = read_value<uint32_t>(ptr);
    } else if (key == "tokenizer.chat_template" && type == GGUF_TYPE_STRING) {
      chat_template_ = read_string(ptr);
    } else {
      skip_metadata_value(ptr, type); // Skip payloads we don't immediately need
    }
  }

  // 3. Parse Tensor Metadata
  for (uint64_t i = 0; i < header.tensor_count; ++i) {
    TensorInfo info;
    info.name = read_string(ptr);
    info.n_dims = read_value<uint32_t>(ptr);

    for (uint32_t d = 0; d < info.n_dims; ++d) {
      info.shape.push_back(read_value<uint64_t>(ptr));
    }

    info.type = static_cast<ggml_type>(read_value<uint32_t>(ptr));
    info.offset = read_value<uint64_t>(ptr);

    tensors_[info.name] = info;
  }

  // 4. Calculate Global Data Alignment
  uint64_t metadata_size = ptr - static_cast<const uint8_t *>(file_.data());
  data_offset_ =
      metadata_size + (alignment_ - (metadata_size % alignment_)) % alignment_;

  std::cout << "[INFO] Parsed " << vocab_tokens_.size() << " vocabulary tokens.\n";
  std::cout << "[INFO] BOS Token ID resolved to: " << bos_token_id_ << "\n";
}

void GGUFLoader::print_summary() const {
  std::cout << "\nParsed " << tensors_.size() << " tensors.\n";
  std::cout << "Data block begins at offset: " << data_offset_ << " bytes.\n";

  int count = 0;
  for (const auto &[name, info] : tensors_) {
    std::cout << "Tensor: " << name << " | Dims: " << info.n_dims
              << " | Type: " << info.type << "\n";
    if (++count >= 5)
      break;
  }
}

Tensor GGUFLoader::get_tensor(const std::string &name) {
  // 1. Locate the metadata entry for the tensor
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::runtime_error("Tensor weight matching name '" + name +
                             "' not found in GGUF file architecture.");
  }

  const TensorInfo &info = it->second;

  // 2. Perform zero-copy pointer math using base mmap layout
  // Base Pointer + Start of data partition aligned to boundary + Individual
  // relative tensor file offset
  uint8_t *file_base = static_cast<uint8_t *>(const_cast<void *>(file_.data()));
  uint8_t *target_tensor_ptr = file_base + data_offset_ + info.offset;

  // Translate the primitive GGML integer type into our engine's DataType
  DataType tensor_type = map_ggml_type(info.type);

  // Calculate expected size
  size_t num_elements = 1;
  for (auto d : info.shape)
    num_elements *= d;
  size_t block_size = get_block_size(tensor_type);
  size_t type_size = get_type_size(tensor_type);
  size_t total_bytes = (num_elements / block_size) * type_size;

  if (data_offset_ + info.offset + total_bytes > file_.size()) {
    throw std::runtime_error("Tensor '" + name +
                             "' is beyond the end of the file. The GGUF file "
                             "is truncated or corrupted!");
  }

  // 3. Return initialized tensor wrapper pointing precisely to the file page
  return Tensor(info.shape, tensor_type, target_tensor_ptr);
}

bool GGUFLoader::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

} // namespace turbo
