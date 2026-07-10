#include "../include/turbo/tensor/tensor.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

TEST(TensorTest, Initialization) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  Tensor t(data, {2, 3});
  EXPECT_EQ(t.rank(), 2);
  EXPECT_EQ(t.numel(), 6);
  EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorTest, Indexing) {
  std::vector<float> data = {10, 20, 30, 40, 50, 60};
  Tensor t(data, {2, 3});
  EXPECT_EQ(t.at({0, 1}), 20);
  EXPECT_EQ(t.at({1, 2}), 60);
  t.at({1, 2}) = 99;
  EXPECT_EQ(t.at({1, 2}), 99);
}

TEST(TensorTest, BroadcastingAddition) {
  Tensor matrix({1, 2, 3, 4, 5, 6}, {2, 3});
  Tensor bias({10, 20, 30}, {3});
  Tensor result = matrix + bias;
  EXPECT_EQ(result.at({0, 0}), 11);
  EXPECT_EQ(result.at({1, 2}), 36);
}

TEST(TensorTest, ViewManipulations) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  Tensor t(data, {2, 3});

  Tensor flat = t.flatten();
  EXPECT_EQ(flat.rank(), 1);
  EXPECT_EQ(flat.get_shape()[0], 6);

  Tensor reshaped = t.reshape({3, 2});
  EXPECT_EQ(reshaped.rank(), 2);

  Tensor unsqueezed = t.unsqueeze(1);
  EXPECT_EQ(unsqueezed.rank(), 3);

  Tensor squeezed = unsqueezed.squeeze(1);
  EXPECT_EQ(squeezed.rank(), 2);

  Tensor transposed = t.transpose();
  EXPECT_FALSE(transposed.is_contiguous());
  EXPECT_THROW(transposed.flatten(), std::runtime_error);
}

TEST(TensorTest, MatMul) {
  // Test basic 2x3 * 3x2 multiplication
  Tensor A({1, 2, 3, 4, 5, 6}, {2, 3});
  Tensor B({7, 8, 9, 10, 11, 12}, {3, 2});
  Tensor C = A.matmul(B);

  EXPECT_EQ(C.rank(), 2);
  EXPECT_EQ(C.get_shape()[0], 2);
  EXPECT_EQ(C.get_shape()[1], 2);

  // Expected result matrix:
  // [58, 64]
  // [139, 154]
  EXPECT_NEAR(C.at({0, 0}), 58.0f, 1e-5);
  EXPECT_NEAR(C.at({0, 1}), 64.0f, 1e-5);
  EXPECT_NEAR(C.at({1, 0}), 139.0f, 1e-5);
  EXPECT_NEAR(C.at({1, 1}), 154.0f, 1e-5);

  // Test with non-multiple of 8 dimensions (e.g. 3x3 * 3x3)
  Tensor A2({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});
  Tensor B2({9, 8, 7, 6, 5, 4, 3, 2, 1}, {3, 3});
  Tensor C2 = A2.matmul(B2);

  // Expected result:
  // [1*9 + 2*6 + 3*3 = 30, 1*8 + 2*5 + 3*2 = 24, 1*7 + 2*4 + 3*1 = 18]
  // [4*9 + 5*6 + 6*3 = 84, 4*8 + 5*5 + 6*2 = 69, 4*7 + 5*4 + 6*1 = 54]
  // [7*9 + 8*6 + 9*3 = 138, 7*8 + 8*5 + 9*2 = 114, 7*7 + 8*4 + 9*1 = 90]
  EXPECT_NEAR(C2.at({0, 0}), 30.0f, 1e-5);
  EXPECT_NEAR(C2.at({0, 1}), 24.0f, 1e-5);
  EXPECT_NEAR(C2.at({0, 2}), 18.0f, 1e-5);
  EXPECT_NEAR(C2.at({1, 0}), 84.0f, 1e-5);
  EXPECT_NEAR(C2.at({1, 1}), 69.0f, 1e-5);
  EXPECT_NEAR(C2.at({1, 2}), 54.0f, 1e-5);
  EXPECT_NEAR(C2.at({2, 0}), 138.0f, 1e-5);
  EXPECT_NEAR(C2.at({2, 1}), 114.0f, 1e-5);
  EXPECT_NEAR(C2.at({2, 2}), 90.0f, 1e-5);
}