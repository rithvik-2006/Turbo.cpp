#include "../../include/turbo/tensor/tensor.hpp"
#include "quant_math.hpp"
#include <algorithm>
#include <cstddef>
#include <cstring>
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

Tensor::Tensor()
    : storage(std::make_shared<Storage>()), offset(0), shape({}), strides({}) {}

Tensor::Tensor(vector<float> d, vector<size_t> s)
    : storage(make_shared<Storage>(std::move(d))), offset(0),
      shape(std::move(s)) {
  strides = compute_strides(shape);
}

Tensor::Tensor(shared_ptr<Storage> st, size_t off, vector<size_t> s,
               vector<size_t> str)
    : storage(std::move(st)), offset(off), shape(std::move(s)),
      strides(std::move(str)) {}

Tensor::Tensor(const std::vector<size_t> &s, DataType dtype, void *external_ptr)
    : shape(s), offset(0), dtype_(dtype) {
  strides = compute_strides(shape);

  // Calculate total logical elements
  size_t num_elements = 1;
  for (auto d : shape)
    num_elements *= d;

  // Calculate exact physical byte size of the mapped block
  size_t block_size = get_block_size(dtype);
  size_t type_size = get_type_size(dtype);

  if (num_elements % block_size != 0) {
    throw std::invalid_argument("Tensor shape is not divisible by block size.");
  }

  size_t total_bytes = (num_elements / block_size) * type_size;

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
  if (is_contiguous())
    return *this;

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
    throw std::invalid_argument(
        "transpose() without args requires a 2D tensor.");
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

Tensor Tensor::slice(size_t dim, size_t start, size_t end) const {
  if (dim >= shape.size())
    throw std::invalid_argument("Dimension out of range.");
  if (start > end || end > shape[dim])
    throw std::out_of_range("Slice range out of bounds.");

  size_t new_offset = offset + (start * strides[dim]);
  vector<size_t> new_shape = shape;
  new_shape[dim] = end - start;

  return Tensor(storage, new_offset, new_shape, strides);
}

Tensor Tensor::reshape(const vector<size_t> &new_shape) const {
  if (!is_contiguous())
    throw std::runtime_error("Cannot reshape a non-contiguous tensor.");

  size_t new_numel = 1;
  for (size_t s : new_shape)
    new_numel *= s;
  if (new_numel != numel()) {
    std::string err = "Reshape changes total elements. Old shape: [";
    for (size_t s : shape)
      err += std::to_string(s) + ", ";
    err += "] (numel=" + std::to_string(numel()) + ") -> New shape: [";
    for (size_t s : new_shape)
      err += std::to_string(s) + ", ";
    err += "] (new_numel=" + std::to_string(new_numel) + ")";
    throw std::invalid_argument(err);
  }

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
  if (this->is_empty())
    return other;
  if (other.is_empty())
    return *this;

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
                         size_t i_end, size_t k_end, size_t stride_A,
                         size_t inner_stride_A) {
  size_t idx = 0;
  for (size_t i = i0; i < i_end; i++) {
    for (size_t k = k0; k < k_end; k++) {
      packed_A[idx++] = A[i * stride_A + k * inner_stride_A];
    }
  }
}

static void pack_block_B(const float *B, float *packed_B, size_t k0, size_t j0,
                         size_t k_end, size_t j_end, size_t stride_B,
                         size_t inner_stride_B) {
  size_t idx = 0;
  for (size_t k = k0; k < k_end; k++) {
    for (size_t j = j0; j < j_end; j++) {
      packed_B[idx++] = B[k * stride_B + j * inner_stride_B];
    }
  }
}

Tensor Tensor::matmul(const Tensor &other) const {
  if (this->get_shape().size() < 2 || other.get_shape().size() < 2) {
    throw invalid_argument("matmul requires at least 2D tensors");
  }

  size_t rank = this->get_shape().size();

  // Phase 4: Intercept Q8_0 weights and route to Quantized Math
  if (this->dtype_ == DataType::FP32 && other.dtype_ == DataType::Q8_0) {
    // other (W) is expected to be shape [out_features, in_features] (i.e. [N,
    // K]) this (X) is shape [batch*seq_len, in_features] (i.e. [M, K])

    size_t M = 1;
    for (size_t i = 0; i < rank - 1; ++i)
      M *= this->get_shape()[i];
    size_t K = this->get_shape()[rank - 1];

    size_t W_K = other.get_shape()[0];
    size_t W_N = other.get_shape()[1];

    if (K != W_K) {
      throw invalid_argument(
          "Inner dimensions must match for quantized matmul");
    }

    std::vector<size_t> result_shape = this->get_shape();
    result_shape.back() = W_N;

    std::vector<float> zeros(M * W_N, 0.0f);
    Tensor result(zeros, result_shape);

    const float *X_data = this->data_ptr();
    float *Y_data = result.data_ptr();
    const uint8_t *W_data =
        reinterpret_cast<const uint8_t *>(other.storage->data());

    size_t block_size = get_block_size(DataType::Q8_0);
    size_t type_size = get_type_size(DataType::Q8_0);
    size_t row_bytes = (W_K / block_size) * type_size;
    size_t nb = W_K / block_size;

    std::vector<uint8_t> x_quantized(M * nb * type_size);

// Pre-quantize all M rows of X
#pragma omp parallel for
    for (size_t m = 0; m < M; ++m) {
      turbo::quantize_row_q8_0(X_data + m * K,
                               x_quantized.data() + m * nb * type_size, K);
    }

#pragma omp parallel for collapse(2)
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < W_N; ++n) {
        const void *w_row = W_data + n * row_bytes;
        // const void* x_row = x_quantized.data() + m * nb * type_size;
        // Y_data[m * W_N + n] = turbo::vec_dot_q8_0_q8_0_avx2(W_K, w_row,
        // x_row);

        const float *x_fp32 = X_data + m * K;
        Y_data[m * W_N + n] = turbo::vec_dot_q8_0_f32(W_K, w_row, x_fp32);
      }
    }
    return result;
  } else if (this->dtype_ == DataType::FP32 && other.dtype_ == DataType::Q4_0) {
    size_t M = 1;
    for (size_t i = 0; i < rank - 1; ++i)
      M *= this->get_shape()[i];
    size_t K = this->get_shape()[rank - 1];

    size_t W_K = other.get_shape()[0];
    size_t W_N = other.get_shape()[1];

    if (K != W_K) {
      throw invalid_argument(
          "Inner dimensions must match for quantized matmul");
    }

    std::vector<size_t> result_shape = this->get_shape();
    result_shape.back() = W_N;

    std::vector<float> zeros(M * W_N, 0.0f);
    Tensor result(zeros, result_shape);

    const float *X_data = this->data_ptr();
    float *Y_data = result.data_ptr();
    const uint8_t *W_data =
        reinterpret_cast<const uint8_t *>(other.storage->data());

    size_t block_size = get_block_size(DataType::Q4_0);
    size_t type_size = get_type_size(DataType::Q4_0);
    size_t row_bytes = (W_K / block_size) * type_size;
    size_t nb = W_K / block_size;

    // Quantize Once Optimization: Compress FP32 into Q8_0
    size_t q8_type_size = get_type_size(DataType::Q8_0);
    std::vector<uint8_t> x_quantized(M * nb * q8_type_size);

#pragma omp parallel for
    for (size_t m = 0; m < M; ++m) {
      turbo::quantize_row_q8_0(X_data + m * K,
                               x_quantized.data() + m * nb * q8_type_size, K);
    }

#pragma omp parallel for collapse(2)
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < W_N; ++n) {
        const void *w_row = W_data + n * row_bytes;
        const void *x_row = x_quantized.data() + m * nb * q8_type_size;
        Y_data[m * W_N + n] = turbo::vec_dot_q4_0_q8_0_avx2(W_K, w_row, x_row);
      }
    }
    return result;
  }

  if (rank != other.get_shape().size()) {
    throw invalid_argument(
        "matmul requires both tensors to have the same rank");
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

  size_t inner_stride_A = this->strides[rank - 1];
  size_t inner_stride_B = other.strides[rank - 1];
  size_t inner_stride_C = result.strides[rank - 1];

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

      const float *A_ptr =
          reinterpret_cast<float *>(this->storage->data()) + a_offset;
      const float *B_ptr =
          reinterpret_cast<float *>(other.storage->data()) + b_offset;
      float *C_ptr =
          reinterpret_cast<float *>(result.storage->data()) + c_offset;

      alignas(32) float packed_A[T * T];
      alignas(32) float packed_B[T * T];

      size_t i_end = std::min(i0 + T, M);
      size_t block_M = i_end - i0;

      for (size_t k0 = 0; k0 < K; k0 += T) {
        size_t k_end = std::min(k0 + T, K);
        size_t block_K = k_end - k0;

        pack_block_A(A_ptr, packed_A, i0, k0, i_end, k_end, stride_A,
                     inner_stride_A);

        for (size_t j0 = 0; j0 < N; j0 += T) {
          size_t j_end = std::min(j0 + T, N);
          size_t block_N = j_end - j0;

          pack_block_B(B_ptr, packed_B, k0, j0, k_end, j_end, stride_B,
                       inner_stride_B);

          for (size_t i = 0; i < block_M; i++) {
            for (size_t k = 0; k < block_K; k++) {
              float a_val = packed_A[i * block_K + k];
              __m256 a_vec = _mm256_set1_ps(a_val);
              size_t j = 0;

              for (; j + 7 < block_N; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&packed_B[k * block_N + j]);
                __m256 c_vec =
                    _mm256_loadu_ps(&C_ptr[(i0 + i) * stride_C + (j0 + j)]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C_ptr[(i0 + i) * stride_C + (j0 + j)], c_vec);
              }

              for (; j < block_N; j++) {
                C_ptr[(i0 + i) * stride_C + (j0 + j)] +=
                    a_val * packed_B[k * block_N + j];
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
  return reinterpret_cast<float *>(storage->data()) + offset;
}

const float *Tensor::data_ptr() const {
  return reinterpret_cast<float *>(storage->data()) + offset;
}

void *Tensor::data_ptr_raw() const {
  if (dtype_ == DataType::FP32) {
    return reinterpret_cast<void *>(reinterpret_cast<float *>(storage->data()) +
                                    offset);
  } else {
    // Offset for quantized types is typically in elements, but for quantized it
    // might just be 0 If it's not 0, we'd need to know the block size. We
    // assume offset is 0 for weights.
    return reinterpret_cast<void *>(static_cast<uint8_t *>(storage->data()));
  }
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
  return reinterpret_cast<float *>(storage->data())[flat_index];
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
  return reinterpret_cast<float *>(storage->data())[flat_index];
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
        pack_block_A(A, packed_A, i0, k0, i_end, k_end, stride_A, 1);

        for (size_t j0 = 0; j0 < N; j0 += T) {
          size_t j_end = std::min(j0 + T, N);
          size_t block_N = j_end - j0;

          // Pack B block (Thread-safe: writing to private packed_B)
          pack_block_B(B, packed_B, k0, j0, k_end, j_end, stride_B, 1);

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
      cout << reinterpret_cast<float *>(storage->data())[index] << " ";
    }
    cout << "\n";
  }
}
// Helper to convert a flat logical index to multi-dimensional coordinates
std::vector<size_t> compute_coords(size_t logical_index,
                                   const std::vector<size_t> &shape) {
  std::vector<size_t> coords(shape.size());
  for (int i = shape.size() - 1; i >= 0; --i) {
    coords[i] = logical_index % shape[i];
    logical_index /= shape[i];
  }
  return coords;
}

// Helper to convert multi-dimensional coordinates to a physical memory offset
// using strides
size_t compute_physical_offset(const std::vector<size_t> &coords,
                               const std::vector<size_t> &strides,
                               size_t base_offset) {
  size_t offset = base_offset;
  for (size_t i = 0; i < coords.size(); ++i) {
    offset += coords[i] * strides[i];
  }
  return offset;
}

void Tensor::copy_(const Tensor &src) {
  // 1. Shape Validation
  if (this->shape != src.shape) {
    throw std::invalid_argument(
        "Tensor::copy_ failed: Target and source shapes must match exactly.");
  }
  if (this->dtype_ != src.dtype_) {
    throw std::invalid_argument(
        "Tensor::copy_ failed: Target and source data types must match.");
  }

  size_t total_elements = this->numel();
  float *dest_ptr = reinterpret_cast<float *>(this->storage->data());
  const float *src_ptr = reinterpret_cast<const float *>(src.storage->data());

  // 2. Fast Path: Both tensors are contiguous in memory
  if (this->is_contiguous() && src.is_contiguous()) {
    std::memcpy(dest_ptr + this->offset, src_ptr + src.offset,
                total_elements * sizeof(float));
    return;
  }

  // 3. Strided Path: One or both tensors are sliced views (Non-contiguous)
  // We iterate over the logical layout and map to physical memory via strides
  for (size_t i = 0; i < total_elements; ++i) {
    // Find N-dimensional coordinates for the current logical index
    std::vector<size_t> coords = compute_coords(i, this->shape);

    // Map coordinates to physical memory locations
    size_t dest_physical_idx =
        compute_physical_offset(coords, this->strides, this->offset);
    size_t src_physical_idx =
        compute_physical_offset(coords, src.strides, src.offset);

    // Execute the single element copy
    dest_ptr[dest_physical_idx] = src_ptr[src_physical_idx];
  }
}
