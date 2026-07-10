#include "turbo/nn/rope.hpp"
#include <gtest/gtest.h>

TEST(LayerTest, RoPEForwardPass) {
  using namespace turbo;

  // 1. Setup Layer: Head dim 4, Max Seq 10
  nn::RoPE rope_layer(4, 10);

  // 2. Setup Input Tensor: [SeqLen=2, HeadDim=4]
  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f, 4.0f, // Token 0
      1.0f, 2.0f, 3.0f, 4.0f  // Token 1
  };
  Tensor X(input_data, {2, 4});

  // 3. Apply RoPE in-place
  rope_layer.apply_in_place(X, 0); // Offset 0

  // 4. Validate Token 0 (Position 0 should not change)
  EXPECT_FLOAT_EQ(X.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(X.at({0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(X.at({0, 2}), 3.0f);
  EXPECT_FLOAT_EQ(X.at({0, 3}), 4.0f);

  // 5. Validate Token 1 (Position 1 rotation)
  // theta_0 = 10000^(-0/4) = 1.0 -> freq = 1.0
  // theta_1 = 10000^(-2/4) = 0.01 -> freq = 0.01
  // New X0 = 1.0 * cos(1) - 3.0 * sin(1) = 0.5403 - 2.5244 = -1.984
  // New X2 = 1.0 * sin(1) + 3.0 * cos(1) = 0.8414 + 1.6209 = 2.462

  EXPECT_NEAR(X.at({1, 0}), -1.9840f, 1e-3);
  EXPECT_NEAR(X.at({1, 2}), 2.4623f, 1e-3);
}