#include "turbo/nn/linear.hpp"
#include "turbo/nn/swiglu.hpp"
#include "turbo/nn/embedding.hpp"
#include "turbo/nn/rmsnorm.hpp"
#include "turbo/tensor/tensor.hpp"
#include <gtest/gtest.h>

TEST(LayerTest, LinearForwardPass) {
  using namespace turbo;

  // 1. Setup layer: 3 input features, 2 output features
  nn::Linear linear_layer(3, 2);

  // 2. Manually inject weights and biases for deterministic testing
  // Weight shape: [3, 2]
  linear_layer.get_weight().at({0, 0}) = 1.0f;
  linear_layer.get_weight().at({0, 1}) = 2.0f;
  linear_layer.get_weight().at({1, 0}) = 3.0f;
  linear_layer.get_weight().at({1, 1}) = 4.0f;
  linear_layer.get_weight().at({2, 0}) = 5.0f;
  linear_layer.get_weight().at({2, 1}) = 6.0f;

  // Bias shape: [2]
  linear_layer.get_bias().at({0}) = 0.5f;
  linear_layer.get_bias().at({1}) = -0.5f;

  // 3. Create Input Tensor X: Batch size 2, 3 features. Shape: [2, 3]
  std::vector<float> input_data = {1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 1.0f};
  Tensor X(input_data, {2, 3});

  // 4. Run the forward pass
  Tensor Y = linear_layer.forward(X);

  // 5. Validate against expected mathematical output
  // Row 0: [1(1)+1(3)+1(5) + 0.5,  1(2)+1(4)+1(6) - 0.5]  => [9.5, 11.5]
  // Row 1: [2(1)+0(3)+1(5) + 0.5,  2(2)+0(4)+1(6) - 0.5]  => [7.5,  9.5]

  EXPECT_FLOAT_EQ(Y.at({0, 0}), 9.5f);
  EXPECT_FLOAT_EQ(Y.at({0, 1}), 11.5f);
  EXPECT_FLOAT_EQ(Y.at({1, 0}), 7.5f);
  EXPECT_FLOAT_EQ(Y.at({1, 1}), 9.5f);
}

TEST(LayerTest, SwiGLUForwardPass) {
    using namespace turbo;
    
    // 1. Setup Input Tensor (Shape: [1, 4])
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor X(input_data, {1, 4});
    
    // 2. Run Forward Pass
    nn::SwiGLU swiglu_layer;
    Tensor Y = swiglu_layer.forward(X);
    
    // 3. Mathematical Validation
    // The layer splits [1, 2, 3, 4] into X1 = [1, 2] and X2 = [3, 4]
    // SiLU(1.0) = 1.0 / (1.0 + exp(-1.0)) = 0.731058
    // SiLU(2.0) = 2.0 / (1.0 + exp(-2.0)) = 1.761594
    // Expected Output = [0.731058 * 3.0, 1.761594 * 4.0] = [2.19317, 7.04637]
    
    // Check new shape
    EXPECT_EQ(Y.rank(), 2);
    EXPECT_EQ(Y.get_shape()[0], 1);
    EXPECT_EQ(Y.get_shape()[1], 2); // Hidden dimension must be halved
    
    // Check math (using EXPECT_NEAR for floating point precision drift)
    EXPECT_NEAR(Y.at({0, 0}), 2.19317f, 1e-4);
    EXPECT_NEAR(Y.at({0, 1}), 7.04637f, 1e-4);
}

TEST(LayerTest, SwiGLUInvalidShape) {
    using namespace turbo;
    // SwiGLU requires an even last dimension. An odd dimension should throw.
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
    Tensor X(input_data, {1, 3});
    nn::SwiGLU swiglu_layer;
    
    EXPECT_THROW(swiglu_layer.forward(X), std::invalid_argument);
}

TEST(LayerTest, EmbeddingForwardPass) {
    using namespace turbo;
    
    // 1. Setup Layer: Vocab Size = 3, Embedding Dim = 4
    nn::Embedding emb_layer(3, 4);
    
    // 2. Manually inject predictable weights
    float val = 0.0f;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            emb_layer.get_weight().at({i, j}) = val++;
        }
    }
    
    // 3. Test Decode Phase (Single Token ID) -> Should trigger zero-copy slice
    std::vector<float> single_token_data = {2.0f};
    Tensor single_token(single_token_data, {1}); // Asking for Token ID 2
    Tensor y_single = emb_layer.forward(single_token);
    
    EXPECT_EQ(y_single.rank(), 1);
    EXPECT_EQ(y_single.get_shape()[0], 4);
    EXPECT_FLOAT_EQ(y_single.at({0}), 8.0f); // Starts at 8
    EXPECT_FLOAT_EQ(y_single.at({3}), 11.0f); // Ends at 11
    
    // 4. Test Prefill Phase (Batched Tokens) -> Should allocate 3D tensor
    std::vector<float> batched_tokens_data = {2.0f, 0.0f};
    Tensor batched_tokens(batched_tokens_data, {2}); // Asking for Token IDs 2 and 0
    Tensor y_batch = emb_layer.forward(batched_tokens);
    
    EXPECT_EQ(y_batch.rank(), 3);
    EXPECT_EQ(y_batch.get_shape()[0], 1); // Batch size 1
    EXPECT_EQ(y_batch.get_shape()[1], 2); // Seq len 2
    EXPECT_EQ(y_batch.get_shape()[2], 4); // Embed dim 4
    
    // Check first token in batch (Token ID 2)
    EXPECT_FLOAT_EQ(y_batch.at({0, 0, 0}), 8.0f);
    EXPECT_FLOAT_EQ(y_batch.at({0, 0, 3}), 11.0f);
    
    // Check second token in batch (Token ID 0)
    EXPECT_FLOAT_EQ(y_batch.at({0, 1, 0}), 0.0f);
    EXPECT_FLOAT_EQ(y_batch.at({0, 1, 3}), 3.0f);
}

TEST(LayerTest, EmbeddingOutOfBounds) {
    using namespace turbo;
    nn::Embedding emb_layer(3, 4);
    
    // Asking for Token ID 5 when vocab size is only 3
    std::vector<float> bad_token_data = {5.0f};
    Tensor bad_token(bad_token_data, {1}); 
    
    EXPECT_THROW(emb_layer.forward(bad_token), std::out_of_range);
}

TEST(LayerTest, RMSNormForwardPass) {
    using namespace turbo;
    
    size_t hidden_dim = 4;
    nn::RMSNorm rms_layer(hidden_dim, 1e-5f);
    
    // 1. Setup Input Tensor (Shape: [1, 4])
    // Data: [1.0, 2.0, 3.0, 4.0]
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor X(input_data, {1, 4});
    
    // Let's manually set the weights to something other than 1.0 to test the scaling
    // Weights: [0.5, 1.0, 1.5, 2.0]
    rms_layer.get_weight().at({0}) = 0.5f;
    rms_layer.get_weight().at({1}) = 1.0f;
    rms_layer.get_weight().at({2}) = 1.5f;
    rms_layer.get_weight().at({3}) = 2.0f;

    // 2. Run Forward Pass
    Tensor Y = rms_layer.forward(X);
    
    // 3. Mathematical Validation
    // Sum of squares: 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
    // Mean of squares: 30 / 4 = 7.5
    // RMS: sqrt(7.5 + 1e-5) ≈ 2.738616
    
    // Expected pre-weight values: [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386] 
    //                         = [0.3651, 0.7303, 1.0954, 1.4606]
    // Expected post-weight values (multiply by [0.5, 1.0, 1.5, 2.0]):
    //                         = [0.1825, 0.7303, 1.6431, 2.9212]

    EXPECT_NEAR(Y.at({0, 0}), 0.182574f, 1e-4);
    EXPECT_NEAR(Y.at({0, 1}), 0.730296f, 1e-4);
    EXPECT_NEAR(Y.at({0, 2}), 1.643167f, 1e-4);
    EXPECT_NEAR(Y.at({0, 3}), 2.921186f, 1e-4);
}