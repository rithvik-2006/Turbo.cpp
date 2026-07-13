<div align="center">
  
# 🚀 Turbo.cpp
**A High-Performance, Zero-Dependency LLM Inference Engine in Pure C++**

[![C++](https://img.shields.io/badge/C++-17%2B-blue.svg?style=for-the-badge&logo=c%2B%2B)](#)
[![Architecture](https://img.shields.io/badge/Architecture-x86__64%20%7C%20AVX2-lightgrey.svg?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](#)
[![Models](https://img.shields.io/badge/Models-Llama--3.2%20%7C%20Mistral-orange.svg?style=for-the-badge)](#)

*Turbo.cpp is a sequence-aware, memory-efficient Large Language Model inference pipeline built entirely from scratch. It bridges the gap between raw hardware performance and modern AI architectures.*

</div>

---

## 💡 Overview

Turbo.cpp is not just a wrapper—it is a custom-built, from-scratch Transformer engine. Designed to run state-of-the-art models like **Llama-3.2**, it bypasses heavy frameworks (like PyTorch or TensorFlow) to interact directly with hardware. 

By implementing **POSIX `mmap()` zero-copy loading**, **AVX2 SIMD intrinsics**, and a **Dynamic KV-Cache**, Turbo.cpp proves that production-grade AI inference can be achieved with pure C++ and rigorous memory management.

---

## ⚡ Core Systems Optimizations (The Engine Room)

To achieve maximum inference speed without OOM (Out Of Memory) crashes, the engine relies on three major systems engineering pillars:

* 💾 **Zero-Copy Memory Mapping:** The massive `.gguf` weight files are mapped directly from disk to virtual memory using `mmap()`. Billions of weights are never duplicated into RAM, resulting in virtually instantaneous model loading.
* 🧮 **AVX2 SIMD & Batched Kernels:** Math is routed to OpenMP parallelized loops (`#pragma omp parallel for collapse(2)`). In the innermost loops, `immintrin.h` intrinsics (`_mm256_*`) load 256 bits of memory at a time, calculating 8 floating-point (or 32 `INT8`) dot-products in a single CPU cycle.
* 📦 **Q8_0 & Q4_0 Quantization:** Full support for loading and executing 8-bit and 4-bit block-quantized weights, drastically reducing memory bandwidth bottlenecks while preserving model accuracy.

---

## 🧠 Architectural Blueprint

### 1. Tokenization & Embedding
* **BPE Tokenizer (`src/core/tokenizer.cpp`)**: A robust Byte-Pair Encoding tokenizer that maps UTF-8 strings to byte fallbacks. It dynamically parses CONTROL tokens (`<|begin_of_text|>`, `<|eot_id|>`) without breaking them apart, ensuring sequence-aware generation halts.
* **Embedding Lookup (`src/nn/embedding.cpp`)**: Extracts dense vectors from a `[vocab_size, hidden_dim]` lookup table. Sequence arrays are seamlessly unsqueezed into `[1, seq_len, hidden_dim]` contiguous tensors.

### 2. Transformer Decoder Stack (`src/nn/transformer_block.cpp`)
The engine dynamically layers $L$ Transformer blocks, weaving together advanced LLM mechanics:
* **Grouped Query Attention (GQA)**: To optimize memory, Llama-3's fewer Key (K) and Value (V) heads are physically shared across multiple Query (Q) heads during dot-products. Includes global causal masking (`if (k_pos > q_pos + offset)`) for batch-processing integrity.
* **Rotary Positional Encodings (RoPE)**: Replaces absolute embeddings with precomputed `sin`/`cos` caches. Applies the GPT-J "rotary-half" mathematical sequence dynamically over the batch based on absolute global indices.
* **SwiGLU FFN**: Implements the `SiLU` (Sigmoid Linear Unit) non-linearity. Projects inputs into massive intermediate dimensions via `gate` and `up` matrices before applying a `down` projection compression.
* **RMSNorm & Residuals**: Utilizes Root Mean Square Normalization to scale variance (bypassing mean-shifting for speed) and leverages zero-copy overloaded C++ operators for residual gradient flow (`h = input + attn_out`).

### 3. State Management & Output Generation
* **Dynamic KV-Cache (`src/core/kv_cache.hpp`)**: A persistent memory arena storing past `K` and `V` tokens. Uses zero-copy slice operations (`K_target_view.copy_()`) to inject new tokens into memory offsets, completely avoiding redundant re-calculations.
* **The "Silver Bullet" Output Layer**: Instead of passing every prompt token into the massive $2048 \times 128256$ vocabulary projection matrix, Turbo.cpp dynamically slices only the *final* generated hidden state vector (`last_token_x.contiguous()`). **Result:** Skips billions of wasted FLOPs during the prefill phase.

### 4. Sampling Strategies (`src/core/sampler.cpp`)
* **Greedy Decoding:** Ultra-fast `argmax` selection for highest-probability tokens.
* **Top-P (Nucleus) Sampling:** Sorts logits, applies softmax distributions, and samples dynamically from the highest-probability "nucleus" to support adjustable temperatures.

---

## 🛠️ Tech Stack & Skills Demonstrated
* **Language:** C++17 / C++20
* **Compute:** SIMD (AVX2 Intrinsics), OpenMP Multithreading
* **Memory Management:** POSIX `mmap()`, Custom Tensor/Storage architecture, Pointer arithmetic, Zero-copy memory views
* **Machine Learning:** LLM Architectures (Llama-3, Mistral), GGUF Parsing, RoPE, GQA, BPE Tokenization
* **Build System:** CMake, Make

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/rithvik-2006/Turbo.cpp.git
cd Turbo.cpp

# 2. Build the project using CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 3. Run the interactive chat CLI (Requires a downloaded .gguf model)
./turbo_chat ../models/llama-3.2-1B-instruct.q8_0.gguf 0.7
```

## 🧪 Testing

Run the GoogleTest validation suite to ensure mathematical and memory integrity:
```bash
cd build
ctest --output-on-failure
```
