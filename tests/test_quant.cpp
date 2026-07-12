#include <gtest/gtest.h>
#include "turbo/tensor/quant_types.hpp"
#include <cmath>

#include "../src/tensor/quant_math.hpp"

TEST(Quantization, ScalarDotProductQ8_0) {
    // 1. Setup a single Q8_0 block (32 elements)
    turbo::block_q8_0 weight_block;
    
    // FP16 scale represents 0.5f in FP32
    weight_block.d = turbo::convert_f32_to_f16(0.5f); 
    
    // Fill the weights with 1s and -1s
    for (int i = 0; i < QK8_0; i++) {
        weight_block.qs[i] = (i % 2 == 0) ? 2 : -2; 
    }

    // 2. Setup the FP32 activations
    float activations[QK8_0];
    for (int i = 0; i < QK8_0; i++) {
        activations[i] = 1.0f; // Uniform activations
    }

    // 3. Compute the dot product
    // Math: 16 evens (2 * 0.5 * 1.0 = 1.0), 16 odds (-2 * 0.5 * 1.0 = -1.0)
    // Total sum should be exactly 0.0f
    float result = turbo::vec_dot_q8_0_f32(QK8_0, &weight_block, activations);

    // 4. Assert correctness
    EXPECT_NEAR(result, 0.0f, 1e-5f);
}

TEST(Quantization, ScalarDotProductQ4_0) {
    // 1. Setup a single Q4_0 block and Q8_0 activation block
    turbo::block_q4_0 weight_block;
    turbo::block_q8_0 act_block;
    
    // FP16 scale represents 0.5f in FP32
    weight_block.d = turbo::convert_f32_to_f16(0.5f);
    act_block.d = turbo::convert_f32_to_f16(1.0f);
    
    // 2. Fill the weights: symmetric pattern
    // Each byte holds two 4-bit values.
    // Let's set lower nibble to 10 (10 - 8 = 2)
    // Let's set upper nibble to 6  (6 - 8 = -2)
    // Packed byte = (6 << 4) | 10 = 0x6A
    for (int i = 0; i < QK4_0 / 2; i++) {
        weight_block.qs[i] = 0x6A;
    }

    // 3. Setup the Q8_0 activations (all 1s)
    for (int i = 0; i < QK8_0; i++) {
        act_block.qs[i] = 1;
    }

    // 4. Compute the dot product
    // Math: 16 lower nibbles (2 * 1 * 0.5 = 1.0) -> +16.0
    // Math: 16 upper nibbles (-2 * 1 * 0.5 = -1.0) -> -16.0
    // Total sum should be exactly 0.0f
    float result = turbo::vec_dot_q4_0_q8_0(QK8_0, &weight_block, &act_block);

    // 5. Assert correctness
    EXPECT_NEAR(result, 0.0f, 1e-5f);
}
