#pragma once
#include "turbo/tensor/quant_types.hpp"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <immintrin.h>
#include <algorithm>

namespace turbo {

    inline float fp16_to_fp32(uint16_t h) {
        return _cvtsh_ss(h);
    }

    // Computes the dot product of a Q8_0 quantized vector and an FP32 vector
    inline float vec_dot_q8_0_f32(const int n, const void* v_x, const float* y) {
        // n must be a multiple of the block size (32)
        const int nb = n / QK8_0; 
        
        // Cast the raw void* to our structured Q8_0 blocks
        const block_q8_0* x = static_cast<const block_q8_0*>(v_x);
        
        float sum = 0.0f;

        // Iterate over each block of 32 weights
        for (int i = 0; i < nb; i++) {
            // 1. Extract the FP16 scale for this specific block
            float d = fp16_to_fp32(x[i].d);
            
            float block_sum = 0.0f;
            
            // 2. Iterate over the 32 quantized integers in the block
            for (int j = 0; j < QK8_0; j++) {
                // Dequantize the weight: (int8_t weight) * scale
                float weight_f32 = x[i].qs[j] * d;
                
                // Multiply by the corresponding FP32 activation and accumulate
                block_sum += weight_f32 * y[i * QK8_0 + j];
            }
            sum += block_sum;
        }
        
        return sum;
    }

    // Helper: Fast Convert FP32 float to FP16 uint16_t using F16C hardware intrinsic
    inline uint16_t convert_f32_to_f16(float f) {
        return _cvtss_sh(f, 0);
    }

    // Quantizes a float array into Q8_0 blocks dynamically
    inline void quantize_row_q8_0(const float* x, void* v_y, int k) {
        const int nb = k / QK8_0;
        block_q8_0* y = static_cast<block_q8_0*>(v_y);

        for (int i = 0; i < nb; i++) {
            float max_val = 0.0f;
            
            // Find absolute max in the block of 32
            for (int j = 0; j < QK8_0; j++) {
                max_val = std::max(max_val, std::abs(x[i * QK8_0 + j]));
            }

            // Compute scale
            float d = max_val / 127.0f;
            y[i].d = convert_f32_to_f16(d);
            float id = d ? 1.0f / d : 0.0f;

            // Quantize to int8
            for (int j = 0; j < QK8_0; j++) {
                y[i].qs[j] = std::round(x[i * QK8_0 + j] * id);
            }
        }
    }

    // Vectorized Dot Product for AVX2
    inline float vec_dot_q8_0_q8_0_avx2(const int n, const void* v_x, const void* v_y) {
        const int nb = n / QK8_0;
        const block_q8_0* x = static_cast<const block_q8_0*>(v_x);
        const block_q8_0* y = static_cast<const block_q8_0*>(v_y);

        // AVX2 accumulator for the final sum
        __m256 acc = _mm256_setzero_ps();

        for (int i = 0; i < nb; i++) {
            // Load the scales (d_x and d_y), convert to FP32, and multiply them
            float d = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
            __m256 vd = _mm256_set1_ps(d);

            // Load 32 bytes (INT8) from weights and activations
            __m256i bx = _mm256_loadu_si256((const __m256i*)x[i].qs);
            __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

            // Multiply INT8 * INT8 and horizontally add into INT16 using _mm256_maddubs_epi16
            // Note: This requires a specific byte layout and sign extension in AVX2. 
            // For standard AVX2, we sign-extend the 8-bit to 16-bit, then multiply and add:
            __m256i bx_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bx, 0));
            __m256i by_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(by, 0));
            __m256i bx_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bx, 1));
            __m256i by_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(by, 1));

            __m256i p_lo = _mm256_madd_epi16(bx_lo, by_lo); // INT32 sums
            __m256i p_hi = _mm256_madd_epi16(bx_hi, by_hi);

            // Convert integer sums to float and multiply by the combined scale (d)
            __m256 sum_lo = _mm256_cvtepi32_ps(p_lo);
            __m256 sum_hi = _mm256_cvtepi32_ps(p_hi);

            acc = _mm256_fmadd_ps(sum_lo, vd, acc);
            acc = _mm256_fmadd_ps(sum_hi, vd, acc);
        }

        // Horizontal sum of the final AVX register to get the scalar float result
        float result_arr[8];
        _mm256_storeu_ps(result_arr, acc);
        float sum = 0;
        for (int i = 0; i < 8; i++) sum += result_arr[i];

        return sum;
    }

    // Computes the dot product of a Q4_0 (weights) and a Q8_0 (activations) vector
    inline float vec_dot_q4_0_q8_0(const int n, const void* v_x, const void* v_y) {
        // n must be a multiple of 32 (QK8_0 and QK4_0 block sizes)
        const int nb = n / QK8_0; 
        
        // x = 4-bit weights, y = 8-bit activations
        const block_q4_0* x = static_cast<const block_q4_0*>(v_x);
        const block_q8_0* y = static_cast<const block_q8_0*>(v_y);

        float sum = 0.0f;

        for (int i = 0; i < nb; i++) {
            // 1. Extract and combine the scales for the block
            float d_x = fp16_to_fp32(x[i].d);
            float d_y = fp16_to_fp32(y[i].d);
            float d = d_x * d_y; // Combined scale factor

            int block_sum = 0;
            
            // 2. Iterate over the 16 bytes (which contain 32 weights)
            // GGML Q4_0 Layout: 
            // - The first 16 weights are stored in the lower 4 bits of the 16 bytes.
            // - The next 16 weights are stored in the upper 4 bits of the 16 bytes.
            for (int j = 0; j < QK4_0 / 2; j++) {
                uint8_t packed = x[i].qs[j];
                
                // Extract lower 4 bits and subtract 8 to shift range to [-8, 7]
                int8_t w0 = (packed & 0x0F) - 8;
                
                // Extract upper 4 bits and subtract 8 to shift range to [-8, 7]
                int8_t w1 = (packed >> 4) - 8;

                // Multiply by the corresponding 8-bit activations
                // Activation indices: j for the lower nibble, j + 16 for the upper nibble
                block_sum += w0 * y[i].qs[j];
                block_sum += w1 * y[i].qs[j + (QK4_0 / 2)]; 
            }
            
            // 3. Multiply the integer sum by the floating-point scale
            sum += block_sum * d;
        }
        
        return sum;
    }

    // Vectorized Dot Product for AVX2 (Q4_0 weights * Q8_0 activations)
    inline float vec_dot_q4_0_q8_0_avx2(const int n, const void* v_x, const void* v_y) {
        const int nb = n / QK8_0; // Blocks of 32 elements
        
        const block_q4_0* x = static_cast<const block_q4_0*>(v_x);
        const block_q8_0* y = static_cast<const block_q8_0*>(v_y);

        // The master accumulator for our dot product (holds 8 floats)
        __m256 acc = _mm256_setzero_ps();

        // Constant vector containing '8' to sign-extend our 4-bit numbers
        const __m128i sub8 = _mm_set1_epi8(8);
        // Constant bitmask to isolate the lower 4 bits (0x0F)
        const __m128i mask = _mm_set1_epi8(0x0F);

        for (int i = 0; i < nb; i++) {
            // 1. Calculate combined floating-point scale
            float d = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
            __m256 vd = _mm256_set1_ps(d);

            // 2. Load 16 bytes of Q4_0 weights (contains 32 4-bit weights)
            __m128i raw_w = _mm_loadu_si128((const __m128i*)x[i].qs);

            // 3. Unpack Lower 16 Weights: (raw_w & 0x0F) - 8
            __m128i w0 = _mm_and_si128(raw_w, mask);
            w0 = _mm_sub_epi8(w0, sub8);

            // 4. Unpack Upper 16 Weights: ((raw_w >> 4) & 0x0F) - 8
            __m128i w1 = _mm_srli_epi16(raw_w, 4); // Logical shift right
            w1 = _mm_and_si128(w1, mask);
            w1 = _mm_sub_epi8(w1, sub8);

            // 5. Expand 8-bit integers to 16-bit integers so we can multiply safely
            __m256i w0_16 = _mm256_cvtepi8_epi16(w0);
            __m256i w1_16 = _mm256_cvtepi8_epi16(w1);

            // 6. Load 32 bytes of Q8_0 activations
            // We split them into two 16-byte chunks to match our weights
            __m128i act0 = _mm_loadu_si128((const __m128i*)y[i].qs);
            __m128i act1 = _mm_loadu_si128((const __m128i*)(y[i].qs + 16));
            
            __m256i a0_16 = _mm256_cvtepi8_epi16(act0);
            __m256i a1_16 = _mm256_cvtepi8_epi16(act1);

            // 7. SIMD Multiply and Add: INT16 * INT16 -> INT32 sums
            __m256i p0 = _mm256_madd_epi16(w0_16, a0_16);
            __m256i p1 = _mm256_madd_epi16(w1_16, a1_16);

            // 8. Convert INT32 sums to Float32
            __m256 sum0 = _mm256_cvtepi32_ps(p0);
            __m256 sum1 = _mm256_cvtepi32_ps(p1);

            // 9. Scale by FP32 delta and accumulate
            acc = _mm256_fmadd_ps(sum0, vd, acc);
            acc = _mm256_fmadd_ps(sum1, vd, acc);
        }

        // 10. Horizontal add to reduce the 8 floats in the AVX register down to 1 scalar
        float result_arr[8];
        _mm256_storeu_ps(result_arr, acc);
        
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) sum += result_arr[i];

        return sum;
    }
}
