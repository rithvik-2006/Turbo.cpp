#pragma once
#include <cstdint>
#include <type_traits>

// Standard GGML Block Size for Q8_0 and Q4_0 is 32 elements.
#define QK8_0 32
#define QK4_0 32

namespace turbo {

    // ---------------------------------------------------------
    // 8-Bit Quantization Block (Q8_0)
    // ---------------------------------------------------------
    // Layout: [2 Bytes for FP16 Scale] + [32 Bytes for INT8 Weights]
    // Total Size: 34 Bytes representing 32 elements.
    // Compression Ratio: ~4x smaller than FP32 (which takes 128 Bytes).
    struct block_q8_0 {
        uint16_t d;          // FP16 scale (delta)
        int8_t   qs[QK8_0];  // Quantized values (1 byte per weight)
    };

    // Ensure the compiler doesn't pad the struct (must be exactly 34 bytes)
    static_assert(sizeof(block_q8_0) == sizeof(uint16_t) + QK8_0, "wrong q8_0 block size/padding");

    // ---------------------------------------------------------
    // 4-Bit Quantization Block (Q4_0)
    // ---------------------------------------------------------
    // Layout: [2 Bytes for FP16 Scale] + [16 Bytes for 4-bit Weights]
    // Note: Two 4-bit weights are packed into a single 8-bit byte (nibbles).
    // Total Size: 18 Bytes representing 32 elements.
    // Compression Ratio: ~7x smaller than FP32.
    struct block_q4_0 {
        uint16_t d;              // FP16 scale (delta)
        uint8_t  qs[QK4_0 / 2];  // Packed nibbles (2 quants per byte)
    };

    // Ensure the compiler doesn't pad the struct (must be exactly 18 bytes)
    static_assert(sizeof(block_q4_0) == sizeof(uint16_t) + (QK4_0 / 2), "wrong q4_0 block size/padding");

} // namespace turbo
