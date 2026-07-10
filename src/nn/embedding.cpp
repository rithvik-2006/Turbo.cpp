// src/nn/embedding.cpp
#include "../../include/turbo/nn/embedding.hpp"
#include <stdexcept>

namespace turbo {
namespace nn {

Embedding::Embedding(size_t num_embeddings, size_t embedding_dim)
    : num_embeddings(num_embeddings), embedding_dim(embedding_dim),
      weight(std::vector<float>(num_embeddings * embedding_dim, 0.0f), {num_embeddings, embedding_dim}) {
  // In production, weights are populated by the GGUF loader.
}

Tensor Embedding::forward(const Tensor &input_ids) {
  if (input_ids.rank() != 1) {
    throw std::invalid_argument(
        "Embedding layer expects a 1D tensor of Token IDs.");
  }

  // Optimization for Decode Phase (1 token at a time)
  // If we only have 1 token, we can use an O(1) zero-copy slice!
  if (input_ids.get_shape()[0] == 1) {
    // Assume Token IDs are stored as floats in our current architecture
    // (In a complete engine, you would cast from an int32 tensor)
    size_t token_id = static_cast<size_t>(input_ids.at({0}));

    if (token_id >= num_embeddings) {
      throw std::out_of_range("Token ID exceeds vocabulary size.");
    }

    // Return a zero-copy 1D slice of the weight matrix
    return weight.slice(0, token_id);
  }

  // For Prefill Phase (Batched tokens)
  // We allocate a new 3D tensor and copy the respective rows to ensure memory
  // contiguity
  size_t seq_len = input_ids.get_shape()[0];
  std::vector<float> zeros(seq_len * embedding_dim, 0.0f);
  Tensor output(zeros, {1, seq_len, embedding_dim});

  for (size_t i = 0; i < seq_len; ++i) {
    size_t token_id = static_cast<size_t>(input_ids.at({i}));

    if (token_id >= num_embeddings) {
      throw std::out_of_range("Token ID exceeds vocabulary size.");
    }

    // Extract the correct row from the weights
    for (size_t j = 0; j < embedding_dim; ++j) {
      output.at({0, i, j}) = weight.at({token_id, j});
    }
  }

  return output;
}

} // namespace nn
} // namespace turbo