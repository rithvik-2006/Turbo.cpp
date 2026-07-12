#pragma once
#include <cstddef>
#include <memory>
#include <vector>
#include <cstring>

#include "turbo/tensor/quant_types.hpp"

// Supported Data Types
enum class DataType {
    FP32,       // Standard 32-bit Float
    FP16,       // 16-bit Float (Often used for KV cache)
    Q8_0,       // 8-bit Block Quantized
    Q4_0,       // 4-bit Block Quantized
    INT32       // For token IDs
};

// Helper function to determine block properties dynamically
inline size_t get_block_size(DataType type) {
    switch (type) {
        case DataType::Q8_0: return QK8_0;
        case DataType::Q4_0: return QK4_0;
        default: return 1; // FP32, FP16, INT32 have a "block size" of 1
    }
}

inline size_t get_type_size(DataType type) {
    switch (type) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::INT32: return 4;
        case DataType::Q8_0: return sizeof(turbo::block_q8_0);
        case DataType::Q4_0: return sizeof(turbo::block_q4_0);
        default: return 0;
    }
}

// Supported Hardware Devices
enum class Device {
  CPU,
  CUDA, // NVIDIA GPUs
  MPS   // Apple Silicon
};

#include "turbo/loader/gguf_common.hpp"

// The physical memory layer
class Storage {
private:
    uint8_t* data_ = nullptr;
    size_t size_bytes_ = 0;
    bool owns_memory_ = true; // Crucial guard flag

public:
    // Default constructor for empty storage
    Storage() : data_(nullptr), size_bytes_(0), owns_memory_(false) {}

    // Standard allocating constructor
    explicit Storage(size_t size_bytes) 
        : size_bytes_(size_bytes), owns_memory_(true) {
        if (size_bytes_ > 0) {
            data_ = new uint8_t[size_bytes_];
        }
    }
    
    // Backwards compatibility for vector initialization
    explicit Storage(const std::vector<float>& d) 
        : size_bytes_(d.size() * sizeof(float)), owns_memory_(true) {
        if (size_bytes_ > 0) {
            data_ = new uint8_t[size_bytes_];
            std::memcpy(data_, d.data(), size_bytes_);
        }
    }

    // NEW: Zero-Copy constructor for mmap binding
    Storage(void* external_ptr, size_t size_bytes) 
        : data_(static_cast<uint8_t*>(external_ptr)), size_bytes_(size_bytes), owns_memory_(false) {}

    // Delete copy operations to prevent dual management bugs
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    // Destructor: Only deletes if it explicitly owns the underlying pointer
    ~Storage() {
        if (owns_memory_ && data_ != nullptr) {
            delete[] data_;
        }
    }

    uint8_t* data() const { return data_; }
    size_t size_bytes() const { return size_bytes_; }
    bool owns_memory() const { return owns_memory_; }
};

// The metadata layer
class Tensor {
private:
  std::shared_ptr<Storage> storage;
  size_t offset;
  std::vector<size_t> shape;
  std::vector<size_t> strides;

  DataType dtype_ = DataType::FP32;
  Device device_ = Device::CPU;

  static std::vector<size_t> compute_strides(const std::vector<size_t> &shape);
  static std::vector<size_t>
  compute_broadcast_shape(const std::vector<size_t> &s1,
                          const std::vector<size_t> &s2);
  static std::vector<size_t> unravel_index(size_t flat_index,
                                           const std::vector<size_t> &shape);

public:
  // Constructors
  Tensor(); // Default constructor for empty state
  Tensor(std::vector<float> d, std::vector<size_t> s);
  Tensor(std::shared_ptr<Storage> st, size_t off, std::vector<size_t> s,
         std::vector<size_t> str);
         
  // NEW: Zero-copy mmap constructor
  Tensor(const std::vector<size_t>& shape, DataType dtype, void* external_ptr);

  // Metadata Queries
  bool is_empty() const { return storage == nullptr || storage->data() == nullptr; }
  size_t rank() const { return shape.size(); }
  size_t numel() const;
  const std::vector<size_t> &get_shape() const { return shape; }
  DataType dtype() const { return dtype_; }
  Device device() const { return device_; }
  bool is_contiguous() const;
  Tensor contiguous() const;

  // View Manipulators (Zero-Copy)
  Tensor transpose() const;
  Tensor transpose(size_t dim0, size_t dim1) const;
  Tensor slice(size_t dim, size_t index) const;
  Tensor slice(size_t dim, size_t start, size_t end) const;
  Tensor reshape(const std::vector<size_t> &new_shape) const;
  Tensor flatten() const;
  Tensor unsqueeze(size_t dim) const;
  Tensor squeeze(size_t dim) const;
  Tensor broadcast_to(const std::vector<size_t> &target_shape) const;
  void copy_(const Tensor& src);

  // Math Operations
  Tensor operator+(const Tensor &other) const;
  Tensor operator*(float scalar) const;

  // Matrix Multiplication
  Tensor matmul(const Tensor &other) const;
  // Indexing
  float &at(const std::vector<size_t> &indices);
  const float &at(const std::vector<size_t> &indices) const;

  // Data Access
  float *data_ptr();
  const float *data_ptr() const;
  void *data_ptr_raw() const; // For quantized access

  // Utilities
  void print() const;
};

// Raw hardware-optimized math functions
void matmul_threaded_packed_avx2(const float *A, const float *B, float *C,
                                 size_t M, size_t K, size_t N, size_t stride_A,
                                 size_t stride_B, size_t stride_C);