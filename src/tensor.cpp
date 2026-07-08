#include "../include/tensor.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

using namespace std;

// --- Private Helpers ---

vector<size_t> Tensor::compute_strides(const vector<size_t> &shape) {
  size_t rank = shape.size();
  vector<size_t> strides(rank);
  if (rank == 0)
    return strides;

  strides[rank - 1] = 1;
  for (size_t i = rank - 1; i > 0; i--) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

vector<size_t> Tensor::compute_broadcast_shape(const vector<size_t> &s1,
                                               const vector<size_t> &s2) {
  vector<size_t> result;
  int i = s1.size() - 1;
  int j = s2.size() - 1;

  while (i >= 0 || j >= 0) {
    size_t dim1 = (i >= 0) ? s1[i] : 1;
    size_t dim2 = (j >= 0) ? s2[j] : 1;

    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      throw std::invalid_argument("Shapes are not broadcastable.");
    }
    result.insert(result.begin(), std::max(dim1, dim2));
    i--;
    j--;
  }
  return result;
}

vector<size_t> Tensor::unravel_index(size_t flat_index,
                                     const vector<size_t> &shape) {
  vector<size_t> indices(shape.size());
  for (int i = shape.size() - 1; i >= 0; --i) {
    indices[i] = flat_index % shape[i];
    flat_index /= shape[i];
  }
  return indices;
}

// --- Constructors ---

Tensor::Tensor(vector<float> d, vector<size_t> s)
    : storage(make_shared<Storage>(std::move(d))), offset(0),
      shape(std::move(s)) {
  strides = compute_strides(shape);
}

Tensor::Tensor(shared_ptr<Storage> st, size_t off, vector<size_t> s,
               vector<size_t> str)
    : storage(std::move(st)), offset(off), shape(std::move(s)),
      strides(std::move(str)) {}

// --- Metadata Queries ---

size_t Tensor::numel() const {
  if (shape.empty())
    return 0;
  size_t count = 1;
  for (size_t dim : shape)
    count *= dim;
  return count;
}

bool Tensor::is_contiguous() const {
  size_t expected_stride = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    if (strides[i] != expected_stride)
      return false;
    expected_stride *= shape[i];
  }
  return true;
}

// --- View Manipulators ---

Tensor Tensor::transpose() const {
  if (shape.size() != 2)
    throw std::invalid_argument("Transpose currently expects a 2D tensor.");
  vector<size_t> new_shape = {shape[1], shape[0]};
  vector<size_t> new_strides = {strides[1], strides[0]};
  return Tensor(storage, offset, new_shape, new_strides);
}

Tensor Tensor::slice(size_t dim, size_t index) const {
  if (dim >= shape.size())
    throw std::invalid_argument("Dimension out of range.");
  if (index >= shape[dim])
    throw std::out_of_range("Slice index out of bounds.");

  size_t new_offset = offset + (index * strides[dim]);
  vector<size_t> new_shape = shape;
  vector<size_t> new_strides = strides;

  new_shape.erase(new_shape.begin() + dim);
  new_strides.erase(new_strides.begin() + dim);
  return Tensor(storage, new_offset, new_shape, new_strides);
}

Tensor Tensor::reshape(const vector<size_t> &new_shape) const {
  if (!is_contiguous())
    throw std::runtime_error("Cannot reshape a non-contiguous tensor.");

  size_t new_numel = 1;
  for (size_t s : new_shape)
    new_numel *= s;
  if (new_numel != numel())
    throw std::invalid_argument("Reshape changes total elements.");

  return Tensor(storage, offset, new_shape, compute_strides(new_shape));
}

Tensor Tensor::flatten() const { return reshape({numel()}); }

Tensor Tensor::unsqueeze(size_t dim) const {
  if (dim > shape.size())
    throw std::out_of_range("Unsqueeze dimension out of bounds.");
  vector<size_t> new_shape = shape;
  new_shape.insert(new_shape.begin() + dim, 1);

  vector<size_t> new_strides = strides;
  size_t new_stride = (dim < strides.size()) ? strides[dim] : 1;
  new_strides.insert(new_strides.begin() + dim, new_stride);

  return Tensor(storage, offset, new_shape, new_strides);
}

Tensor Tensor::squeeze(size_t dim) const {
  if (dim >= shape.size())
    throw std::out_of_range("Squeeze dimension out of bounds.");
  if (shape[dim] != 1)
    throw std::invalid_argument("Can only squeeze dimensions of size 1.");

  vector<size_t> new_shape = shape;
  vector<size_t> new_strides = strides;

  new_shape.erase(new_shape.begin() + dim);
  new_strides.erase(new_strides.begin() + dim);
  return Tensor(storage, offset, new_shape, new_strides);
}

Tensor Tensor::broadcast_to(const vector<size_t> &target_shape) const {
  size_t target_rank = target_shape.size();
  size_t current_rank = shape.size();
  if (target_rank < current_rank)
    throw std::invalid_argument("Target rank too small.");

  vector<size_t> new_shape = shape;
  vector<size_t> new_strides = strides;
  size_t rank_diff = target_rank - current_rank;

  new_shape.insert(new_shape.begin(), rank_diff, 1);
  new_strides.insert(new_strides.begin(), rank_diff, 0);

  for (size_t i = 0; i < target_rank; i++) {
    if (new_shape[i] != target_shape[i]) {
      if (new_shape[i] == 1) {
        new_shape[i] = target_shape[i];
        new_strides[i] = 0;
      } else {
        throw std::invalid_argument("Shapes are not broadcastable.");
      }
    }
  }
  return Tensor(storage, offset, new_shape, new_strides);
}

// --- Math Operations ---

Tensor Tensor::operator+(const Tensor &other) const {
  vector<size_t> target_shape =
      compute_broadcast_shape(this->shape, other.shape);
  Tensor a_bcast = this->broadcast_to(target_shape);
  Tensor b_bcast = other.broadcast_to(target_shape);

  size_t total_elements = 1;
  for (size_t dim : target_shape)
    total_elements *= dim;
  vector<float> out_data(total_elements, 0.0f);

  Tensor result(out_data, target_shape);
  for (size_t i = 0; i < total_elements; ++i) {
    vector<size_t> idx = unravel_index(i, target_shape);
    result.at(idx) = a_bcast.at(idx) + b_bcast.at(idx);
  }
  return result;
}

// --- Indexing ---

float &Tensor::at(const vector<size_t> &indices) {
  if (indices.size() != shape.size())
    throw std::invalid_argument("Rank mismatch.");
  size_t flat_index = offset;
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i] >= shape[i])
      throw std::out_of_range("Index out of bounds.");
    flat_index += indices[i] * strides[i];
  }
  return storage->data[flat_index];
}

const float &Tensor::at(const vector<size_t> &indices) const {
  if (indices.size() != shape.size())
    throw std::invalid_argument("Rank mismatch.");
  size_t flat_index = offset;
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i] >= shape[i])
      throw std::out_of_range("Index out of bounds.");
    flat_index += indices[i] * strides[i];
  }
  return storage->data[flat_index];
}

// --- Utilities ---

void Tensor::print() const {
  if (shape.size() != 2) {
    cout << "Print currently supports 2D tensors.\n";
    return;
  }
  for (size_t i = 0; i < shape[0]; i++) {
    for (size_t j = 0; j < shape[1]; j++) {
      size_t index = offset + (i * strides[0]) + (j * strides[1]);
      cout << storage->data[index] << " ";
    }
    cout << "\n";
  }
}