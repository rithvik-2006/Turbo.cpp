#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "turbo/loader/mmap_file.hpp"
#include "turbo/loader/gguf_common.hpp"
#include "turbo/tensor/tensor.hpp"

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
    uint32_t alignment_ = 32; // Default GGUF alignment
    uint64_t data_offset_ = 0; // Where the actual weight bytes begin
    
    // Extracted architecture data
    std::unordered_map<std::string, TensorInfo> tensors_;

    // Helper methods for safe pointer traversal
    template <typename T>
    T read_value(const uint8_t*& ptr);
    std::string read_string(const uint8_t*& ptr);
    void skip_metadata_value(const uint8_t*& ptr, uint32_t type);

public:
    explicit GGUFLoader(const std::string& filepath);
    
    void parse();
    void print_summary() const;

    // NEW API
    Tensor get_tensor(const std::string& name);
};

} // namespace turbo
