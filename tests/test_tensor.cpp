#include "turbo/nn/softmax.hpp"
#include <cmath>
#include <gtest/gtest.h>

// Helper function to check if two floats are close enough (handles floating
// point drift)
bool is_close(float a, float b, float epsilon = 1e-4) {
  return std::abs(a - b) < epsilon;
}

TEST(LayerTest, NumericallyStableSoftmax) {
  using namespace turbo;

  // 1. Setup a dummy input (Batch Size 2, Vocab Size 3)
  // We intentionally include a large number (100.0) to test numerical
  // stability. Naive softmax would return NaN here.
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 10.0f, 100.0f, 10.0f};
  std::vector<size_t> shape = {2, 3};
  std::vector<size_t> strides = {3, 1}; // Standard row-major contiguous strides

  // Create the tensor
  Tensor logits(input_data, shape);

  // 2. Initialize the Layer and run the forward pass
  nn::Softmax softmax_layer(-1);
  Tensor probabilities = softmax_layer.forward(logits);

  // 3. Define the Expected Output (Pre-calculated via PyTorch)
  // PyTorch code: torch.softmax(torch.tensor([[1., 2., 3.], [10., 100., 10.]]),
  // dim=-1)
  std::vector<float> expected = {
      0.0900f, 0.2447f, 0.6652f, // Row 0
      0.0000f, 1.0000f, 0.0000f  // Row 1 (100 dominates completely)
  };

  // 4. Validate
  for (size_t i = 0; i < expected.size(); ++i) {
    // Calculate flat index since the returned tensor is contiguous
    float actual_val = probabilities.at({i / 3, i % 3});
    EXPECT_TRUE(is_close(actual_val, expected[i]))
        << "Mismatch at index " << i << ": Expected " << expected[i] << ", got "
        << actual_val;
  }
}