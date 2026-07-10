#include "turbo/loader/gguf_loader.hpp"
#include <iostream>
#include <cstring>

namespace turbo {

// --- Pointer Arithmetic Helpers ---

template <typename T>
T GGUFLoader::read_value(const uint8_t*& ptr) {
    T val;
    std::memcpy(&val, ptr, sizeof(T));
    ptr += sizeof(T);
    return val;
}

std::string GGUFLoader::read_string(const uint8_t*& ptr) {
    uint64_t len = read_value<uint64_t>(ptr);
    std::string s(reinterpret_cast<const char*>(ptr), len);
    ptr += len;
    return s;
}

// Skips over dynamic metadata payloads so we don't crash reading the next key
void GGUFLoader::skip_metadata_value(const uint8_t*& ptr, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   case GGUF_TYPE_INT8:   ptr += 1; break;
        case GGUF_TYPE_UINT16:  case GGUF_TYPE_INT16:  ptr += 2; break;
        case GGUF_TYPE_UINT32:  case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: ptr += 4; break;
        case GGUF_TYPE_UINT64:  case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: ptr += 8; break;
        case GGUF_TYPE_BOOL:    ptr += 1; break;
        case GGUF_TYPE_STRING:  read_string(ptr); break;
        case GGUF_TYPE_ARRAY: {
            uint32_t arr_type = read_value<uint32_t>(ptr);
            uint64_t arr_len = read_value<uint64_t>(ptr);
            for (uint64_t i = 0; i < arr_len; ++i) {
                skip_metadata_value(ptr, arr_type);
            }
            break;
        }
        default: throw std::runtime_error("Unknown GGUF metadata type");
    }
}

// --- Core Loader ---

GGUFLoader::GGUFLoader(const std::string& filepath) : file_(filepath) {}

void GGUFLoader::parse() {
    const uint8_t* ptr = static_cast<const uint8_t*>(file_.data());
    
    // 1. Read and validate header
    auto header = read_value<gguf_header_t>(ptr);
    if (header.magic != GGUF_MAGIC) {
        throw std::runtime_error("Invalid GGUF magic bytes. Not a GGUF file.");
    }
    
    std::cout << "GGUF Version: " << header.version << "\n";
    std::cout << "KV Pairs: " << header.kv_count << " | Tensors: " << header.tensor_count << "\n";

    // 2. Parse KV Metadata
    for (uint64_t i = 0; i < header.kv_count; ++i) {
        std::string key = read_string(ptr);
        uint32_t type = read_value<uint32_t>(ptr);
        
        // Extract alignment if present (critical for calculating final tensor offsets)
        if (key == "general.alignment" && type == GGUF_TYPE_UINT32) {
            alignment_ = read_value<uint32_t>(ptr);
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
    uint64_t metadata_size = ptr - static_cast<const uint8_t*>(file_.data());
    data_offset_ = metadata_size + (alignment_ - (metadata_size % alignment_)) % alignment_;
}

void GGUFLoader::print_summary() const {
    std::cout << "\nParsed " << tensors_.size() << " tensors.\n";
    std::cout << "Data block begins at offset: " << data_offset_ << " bytes.\n";
    
    int count = 0;
    for (const auto& [name, info] : tensors_) {
        std::cout << "Tensor: " << name << " | Dims: " << info.n_dims << " | Type: " << info.type << "\n";
        if (++count >= 5) break;
    }
}

Tensor GGUFLoader::get_tensor(const std::string& name) {
    // 1. Locate the metadata entry for the tensor
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor weight matching name '" + name + "' not found in GGUF file architecture.");
    }
    
    const TensorInfo& info = it->second;

    // 2. Perform zero-copy pointer math using base mmap layout
    // Base Pointer + Start of data partition aligned to boundary + Individual relative tensor file offset
    uint8_t* file_base = static_cast<uint8_t*>(const_cast<void*>(file_.data()));
    uint8_t* target_tensor_ptr = file_base + data_offset_ + info.offset;

    // 3. Return initialized tensor wrapper pointing precisely to the file page
    return Tensor(info.shape, info.type, target_tensor_ptr);
}

} // namespace turbo
