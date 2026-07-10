// include/turbo/nn/rmsnorm.hpp
#pragma once
#include "layer.hpp"
#include "../tensor/tensor.hpp"

namespace turbo {
namespace nn {

class RMSNorm : public Layer {
private:
    Tensor weight;
    float epsilon;
    size_t hidden_size;

public:
    // hidden_size defines the last dimension, epsilon prevents div by zero
    RMSNorm(size_t hidden_size, float epsilon = 1e-5f);
    
    // The execution contract
    Tensor forward(const Tensor& input) override;

    // Getter for weight initialization/loading
    Tensor& get_weight() { return weight; }
};

} // namespace nn
} // namespace turbo