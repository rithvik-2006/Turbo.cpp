#include "../../include/turbo/tensor/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <stdexcept>
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

Tensor::Tensor(const std::vector<size_t>& s, turbo::ggml_type dtype, void* external_ptr)
    : shape(s), offset(0) {
    strides = compute_strides(shape);
    
    // For now we assume Float32 internally if we are just testing pointer mapping,
    // though dtype would dictate actual size in a full impl
    size_t total_bytes = calculate_size_bytes(shape, dtype);
    
    storage = std::make_shared<Storage>(external_ptr, total_bytes);
}

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

Tensor Tensor::contiguous() const {
  if (is_contiguous()) return *this;
  
  std::vector<float> new_data(numel(), 0.0f);
  Tensor result(new_data, shape);
  
  float *dst = result.data_ptr();
  size_t count = numel();
  for (size_t i = 0; i < count; ++i) {
    std::vector<size_t> idx = unravel_index(i, shape);
    dst[i] = this->at(idx);
  }
  return result;
}

// --- View Manipulators ---

Tensor Tensor::transpose() const {
  if (rank() != 2)
    throw std::invalid_argument("transpose() without args requires a 2D tensor.");
  return transpose(0, 1);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
  if (dim0 >= rank() || dim1 >= rank()) {
    throw std::out_of_range("Transpose dimensions out of bounds");
  }
  std::vector<size_t> new_shape = shape;
  std::vector<size_t> new_strides = strides;
  std::swap(new_shape[dim0], new_shape[dim1]);
  std::swap(new_strides[dim0], new_strides[dim1]);
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

Tensor Tensor::operator*(float scalar) const {
  size_t total_elements = numel();
  vector<float> out_data(total_elements, 0.0f);
  Tensor result(out_data, shape);
  
  for (size_t i = 0; i < total_elements; ++i) {
    vector<size_t> idx = unravel_index(i, shape);
    result.at(idx) = this->at(idx) * scalar;
  }
  return result;
}

// --- Static Packing Helpers for Stage 4 ---

static void pack_block_A(const float *A, float *packed_A, size_t i0, size_t k0,
                         size_t i_end, size_t k_end, size_t stride_A) {
  size_t idx = 0;
  for (size_t i = i0; i < i_end; i++) {
    for (size_t k = k0; k < k_end; k++) {
      packed_A[idx++] = A[i * stride_A + k];
    }
  }
}

static void pack_block_B(const float *B, float *packed_B, size_t k0, size_t j0,
                         size_t k_end, size_t j_end, size_t stride_B) {
  size_t idx = 0;
  for (size_t k = k0; k < k_end; k++) {
    for (size_t j = j0; j < j_end; j++) {
      packed_B[idx++] = B[k * stride_B + j];
    }
  }
}

Tensor Tensor::matmul(const Tensor &other) const {
  if (this->get_shape().size() < 2 || other.get_shape().size() < 2) {
    throw invalid_argument("matmul requires at least 2D tensors");
  }

  size_t rank = this->get_shape().size();
  if (rank != other.get_shape().size()) {
    throw invalid_argument("matmul requires both tensors to have the same rank");
  }

  size_t batch_size = 1;
  vector<size_t> result_shape;
  for (size_t i = 0; i < rank - 2; ++i) {
    if (this->get_shape()[i] != other.get_shape()[i]) {
      throw invalid_argument("Batch dimensions must match");
    }
    batch_size *= this->get_shape()[i];
    result_shape.push_back(this->get_shape()[i]);
  }

  size_t M = this->get_shape()[rank - 2];
  size_t K = this->get_shape()[rank - 1];
  size_t N = other.get_shape()[rank - 1];

  if (K != other.get_shape()[rank - 2]) {
    throw invalid_argument("Inner dimensions must match");
  }

  result_shape.push_back(M);
  result_shape.push_back(N);

  vector<float> zeros(batch_size * M * N, 0.0f);
  Tensor result(zeros, result_shape);

  size_t stride_A = this->strides[rank - 2];
  size_t stride_B = other.strides[rank - 2];
  size_t stride_C = result.strides[rank - 2];

  const size_t T = 64;

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t i0 = 0; i0 < M; i0 += T) {
      // Calculate batch offset
      size_t a_offset = this->offset;
      size_t b_offset = other.offset;
      size_t c_offset = result.offset;

      size_t temp_b = b;
      for (int d = rank - 3; d >= 0; --d) {
        size_t idx = temp_b % this->get_shape()[d];
        temp_b /= this->get_shape()[d];
        a_offset += idx * this->strides[d];
        b_offset += idx * other.strides[d];
        c_offset += idx * result.strides[d];
      }

      const float *A_ptr = reinterpret_cast<float*>(this->storage->data()) + a_offset;
      const float *B_ptr = reinterpret_cast<float*>(other.storage->data()) + b_offset;
      float *C_ptr = reinterpret_cast<float*>(result.storage->data()) + c_offset;

      alignas(32) float packed_A[T * T];
      alignas(32) float packed_B[T * T];

      size_t i_end = std::min(i0 + T, M);
      size_t block_M = i_end - i0;

      for (size_t k0 = 0; k0 < K; k0 += T) {
        size_t k_end = std::min(k0 + T, K);
        size_t block_K = k_end - k0;

        pack_block_A(A_ptr, packed_A, i0, k0, i_end, k_end, stride_A);

        for (size_t j0 = 0; j0 < N; j0 += T) {
          size_t j_end = std::min(j0 + T, N);
          size_t block_N = j_end - j0;

          pack_block_B(B_ptr, packed_B, k0, j0, k_end, j_end, stride_B);

          for (size_t i = 0; i < block_M; i++) {
            for (size_t k = 0; k < block_K; k++) {
              float a_val = packed_A[i * block_K + k];
              __m256 a_vec = _mm256_set1_ps(a_val);
              size_t j = 0;

              for (; j + 7 < block_N; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&packed_B[k * block_N + j]);
                __m256 c_vec = _mm256_loadu_ps(&C_ptr[(i0 + i) * stride_C + (j0 + j)]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C_ptr[(i0 + i) * stride_C + (j0 + j)], c_vec);
              }

              for (; j < block_N; j++) {
                C_ptr[(i0 + i) * stride_C + (j0 + j)] += a_val * packed_B[k * block_N + j];
              }
            }
          }
        }
      }
    }
  }

  return result;
}
// --- Indexing ---

float *Tensor::data_ptr() {
  return reinterpret_cast<float*>(storage->data()) + offset;
}

const float *Tensor::data_ptr() const {
  return reinterpret_cast<float*>(storage->data()) + offset;
}

float &Tensor::at(const vector<size_t> &indices) {
  if (indices.size() != shape.size())
    throw std::invalid_argument("Rank mismatch.");
  size_t flat_index = offset;
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i] >= shape[i])
      throw std::out_of_range("Index out of bounds.");
    flat_index += indices[i] * strides[i];
  }
  return reinterpret_cast<float*>(storage->data())[flat_index];
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
  return reinterpret_cast<float*>(storage->data())[flat_index];
}

// --- Multithreading ---
void matmul_threaded_packed_avx2(const float *A, const float *B, float *C,
                                 size_t M, size_t K, size_t N, size_t stride_A,
                                 size_t stride_B, size_t stride_C) {
  const size_t T = 64;

#pragma omp parallel
  {
    alignas(32) float packed_A[T * T];
    alignas(32) float packed_B[T * T];

#pragma omp for schedule(dynamic)
    for (size_t i0 = 0; i0 < M; i0 += T) {
      size_t i_end = std::min(i0 + T, M);
      size_t block_M = i_end - i0;

      for (size_t k0 = 0; k0 < K; k0 += T) {
        size_t k_end = std::min(k0 + T, K);
        size_t block_K = k_end - k0;

        // Pack A block (Thread-safe: writing to private packed_A)
        pack_block_A(A, packed_A, i0, k0, i_end, k_end, stride_A);

        for (size_t j0 = 0; j0 < N; j0 += T) {
          size_t j_end = std::min(j0 + T, N);
          size_t block_N = j_end - j0;

          // Pack B block (Thread-safe: writing to private packed_B)
          pack_block_B(B, packed_B, k0, j0, k_end, j_end, stride_B);

          // --- INNER LOOP: AVX2 Math strictly on Packed Buffers ---
          for (size_t i = 0; i < block_M; i++) {
            for (size_t k = 0; k < block_K; k++) {

              float a_val = packed_A[i * block_K + k];
              __m256 a_vec = _mm256_set1_ps(a_val);

              size_t j = 0;

              for (; j + 7 < block_N; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&packed_B[k * block_N + j]);
                // Safe write to C: Every thread has a unique 'i0',
                // meaning they write to completely disjoint memory addresses.
                __m256 c_vec =
                    _mm256_loadu_ps(&C[(i0 + i) * stride_C + (j0 + j)]);

                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C[(i0 + i) * stride_C + (j0 + j)], c_vec);
              }

              for (; j < block_N; j++) {
                C[(i0 + i) * stride_C + (j0 + j)] +=
                    a_val * packed_B[k * block_N + j];
              }
            }
          }
        }
      }
    }
  }
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
      cout << reinterpret_cast<float*>(storage->data())[index] << " ";
    }
    cout << "\n";
  }
}