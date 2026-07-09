#pragma once
#include <cstddef>
#include <memory>
#include <vector>

// Supported Data Types
enum class DataType {
  Float32,
  Float16, // For mixed precision
  Int8,    // For quantized weights
  Int32    // For token IDs
};

// Supported Hardware Devices
enum class Device {
  CPU,
  CUDA, // NVIDIA GPUs
  MPS   // Apple Silicon
};

// The physical memory layer
struct Storage {
  std::vector<float> data;
  Storage(std::vector<float> d) : data(std::move(d)) {}
};

// The metadata layer
class Tensor {
private:
  std::shared_ptr<Storage> storage;
  size_t offset;
  std::vector<size_t> shape;
  std::vector<size_t> strides;

  DataType dtype_ = DataType::Float32;
  Device device_ = Device::CPU;

  static std::vector<size_t> compute_strides(const std::vector<size_t> &shape);
  static std::vector<size_t>
  compute_broadcast_shape(const std::vector<size_t> &s1,
                          const std::vector<size_t> &s2);
  static std::vector<size_t> unravel_index(size_t flat_index,
                                           const std::vector<size_t> &shape);

public:
  // Constructors
  Tensor(std::vector<float> d, std::vector<size_t> s);
  Tensor(std::shared_ptr<Storage> st, size_t off, std::vector<size_t> s,
         std::vector<size_t> str);

  // Metadata Queries
  size_t rank() const { return shape.size(); }
  size_t numel() const;
  const std::vector<size_t> &get_shape() const { return shape; }
  DataType dtype() const { return dtype_; }
  Device device() const { return device_; }
  bool is_contiguous() const;

  // View Manipulators (Zero-Copy)
  Tensor transpose() const;
  Tensor slice(size_t dim, size_t index) const;
  Tensor reshape(const std::vector<size_t> &new_shape) const;
  Tensor flatten() const;
  Tensor unsqueeze(size_t dim) const;
  Tensor squeeze(size_t dim) const;
  Tensor broadcast_to(const std::vector<size_t> &target_shape) const;

  // Math Operations
  Tensor operator+(const Tensor &other) const;

  // Matrix Multiplication
  Tensor matmul(const Tensor &other) const;
  // Indexing
  float &at(const std::vector<size_t> &indices);
  const float &at(const std::vector<size_t> &indices) const;

  // Utilities
  void print() const;
};