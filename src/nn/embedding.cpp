// src/nn/embedding.cpp
#include "../../include/turbo/nn/embedding.hpp"
#include "../../src/tensor/quant_math.hpp"
#include <stdexcept>

namespace turbo {
namespace nn {

Embedding::Embedding(Tensor w)
    : weight(std::move(w)),
      num_embeddings(weight.get_shape().size() > 1 ? weight.get_shape()[1] : (weight.get_shape().size() > 0 ? weight.get_shape()[0] : 0)),
      embedding_dim(weight.get_shape().size() > 0 ? weight.get_shape()[0] : 0) {
    // In GGUF, token_embd.weight is [hidden_dim, vocab_size] where shape[0]=hidden_dim, shape[1]=vocab_size
}

Embedding::Embedding(size_t num_embeddings, size_t embedding_dim)
    : num_embeddings(num_embeddings), embedding_dim(embedding_dim),
      weight(std::vector<float>(num_embeddings * embedding_dim, 0.0f), {embedding_dim, num_embeddings}) {
  // In production, weights are populated by the GGUF loader.
}

Tensor Embedding::forward(const Tensor &input_ids) {
  if (input_ids.rank() != 1) {
    throw std::invalid_argument(
        "Embedding layer expects a 1D tensor of Token IDs.");
  }

  // Optimization for Decode Phase (1 token at a time)
  if (input_ids.get_shape()[0] == 1) {
    size_t token_id = static_cast<size_t>(input_ids.at({0}));

    if (token_id >= num_embeddings) {
      throw std::out_of_range("Token ID " + std::to_string(token_id) + " exceeds vocabulary size " + std::to_string(num_embeddings));
    }

    // We must copy the token embedding because it is likely strided (non-contiguous)
    // in the weight matrix if shape is [embedding_dim, num_embeddings].
    std::vector<float> output_data(embedding_dim);
    
    if (weight.dtype() == DataType::FP32) {
        Tensor emb = weight.slice(1, token_id);
        for (size_t d = 0; d < embedding_dim; ++d) {
            output_data[d] = emb.at({d});
        }
    } else if (weight.dtype() == DataType::Q8_0) {
        size_t row_bytes = (embedding_dim / 32) * sizeof(block_q8_0);
        const uint8_t* compressed_row = static_cast<const uint8_t*>(weight.data_ptr_raw()) + token_id * row_bytes;
        
        turbo::dequantize_row_q8_0(compressed_row, output_data.data(), embedding_dim);
    } else if (weight.dtype() == DataType::Q4_0) {
        size_t row_bytes = (embedding_dim / 32) * 18;
        const uint8_t* row_ptr = static_cast<const uint8_t*>(weight.data_ptr_raw()) + token_id * row_bytes;
        
        const block_q4_0* x = reinterpret_cast<const block_q4_0*>(row_ptr);
        int nb = embedding_dim / 32;
        for (int b = 0; b < nb; b++) {
            float d = fp16_to_fp32(x[b].d);
            for (int j = 0; j < 16; j++) {
                uint8_t packed = x[b].qs[j];
                output_data[b * 32 + j] = ((packed & 0x0F) - 8) * d;
                output_data[b * 32 + j + 16] = ((packed >> 4) - 8) * d;
            }
        }
    } else {
        throw std::runtime_error("Unsupported embedding dtype");
    }

    return Tensor(output_data, {1, embedding_dim});
  }

  // For Prefill Phase (Batched tokens)
  // We allocate a new 3D tensor and copy the respective rows to ensure memory
  // contiguity
  size_t seq_len = input_ids.get_shape()[0];
  std::vector<float> zeros(seq_len * embedding_dim, 0.0f);
  Tensor output(zeros, {seq_len, embedding_dim});

  for (size_t i = 0; i < seq_len; ++i) {
    size_t token_id = static_cast<size_t>(input_ids.at({i}));

    if (token_id >= num_embeddings) {
      throw std::out_of_range("Token ID " + std::to_string(token_id) + " exceeds vocabulary size " + std::to_string(num_embeddings) + ". Shape: " + std::to_string(weight.get_shape()[0]) + ", " + std::to_string(weight.get_shape()[1]));
    }

    // Extract the correct row from the weights
    if (weight.dtype() == DataType::FP32) {
      for (size_t j = 0; j < embedding_dim; ++j) {
        output.at({i, j}) = weight.at({j, token_id});
      }
    } else if (weight.dtype() == DataType::Q8_0) {
      size_t row_bytes = (embedding_dim / 32) * sizeof(block_q8_0);
      const uint8_t* compressed_row = static_cast<const uint8_t*>(weight.data_ptr_raw()) + token_id * row_bytes;
      
      float* out_ptr = output.data_ptr() + (i * embedding_dim);
      turbo::dequantize_row_q8_0(compressed_row, out_ptr, embedding_dim);
    } else if (weight.dtype() == DataType::Q4_0) {
      size_t row_bytes = (embedding_dim / 32) * 18; // QK4_0 is 32, block size is 18
      const uint8_t* row_ptr = static_cast<const uint8_t*>(weight.data_ptr_raw()) + token_id * row_bytes;
      
      const block_q4_0* x = reinterpret_cast<const block_q4_0*>(row_ptr);
      int nb = embedding_dim / 32;
      for (int b = 0; b < nb; b++) {
          float d = fp16_to_fp32(x[b].d);
          for (int j = 0; j < 16; j++) {
              uint8_t packed = x[b].qs[j];
              int8_t w0 = (packed & 0x0F) - 8;
              int8_t w1 = (packed >> 4) - 8;
              output.at({i, b * 32 + j}) = w0 * d;
              output.at({i, b * 32 + j + 16}) = w1 * d;
          }
      }
    } else {
        throw std::runtime_error("Unsupported embedding dtype");
    }
  }

  return output;
}

} // namespace nn
} // namespace turbo