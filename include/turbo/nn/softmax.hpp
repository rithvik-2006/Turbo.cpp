#pragma once
#include "layer.hpp"

namespace turbo {
namespace nn {
class Softmax : public Layer {
private:
  int dim;

public:
  Softmax(int dim = -1) : dim(dim) {}
  Tensor forward(const Tensor &input) override;
};

} // namespace nn

} // namespace turbo