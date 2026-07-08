#include "../include/tensor.hpp"
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