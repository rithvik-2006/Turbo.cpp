// include/turbo/nn/embedding.hpp
#pragma once
#include "layer.hpp"

namespace turbo {
namespace nn {

class Embedding : public Layer {
private:
  Tensor weight;
  size_t num_embeddings; // Vocabulary size
  size_t embedding_dim;  // Vector dimension

public:
  Embedding(size_t num_embeddings, size_t embedding_dim);

  // forward pass takes a 1D tensor of Token IDs
  Tensor forward(const Tensor &input_ids) override;

  Tensor &get_weight() { return weight; }
};

} // namespace nn
} // namespace turbo