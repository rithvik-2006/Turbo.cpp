#pragma once
#include <cstdint>

namespace turbo {

// GGUF Magic Bytes: "GGUF" in little-endian ASCII (0x46554747)
constexpr uint32_t GGUF_MAGIC = 0x46554747;

#pragma pack(push, 1)
struct gguf_header_t {
    uint32_t magic;          // Must match GGUF_MAGIC
    uint32_t version;        // Currently version 2 or 3
    uint64_t tensor_count;   // Total number of weight tensors in the file
    uint64_t kv_count;       // Total number of metadata Key-Value pairs
};
#pragma pack(pop)

enum gguf_metadata_value_type : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12
};

enum ggml_type : uint32_t {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q8_0 = 8
};

} // namespace turbo
