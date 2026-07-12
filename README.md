# Turbo Engine 🚀

Turbo is a high-performance C++ inference engine designed from the ground up to run state-of-the-art Large Language Models (LLMs) like LLaMA, Mistral, and Gemma natively. 

Built with minimal dependencies, Turbo prioritizes memory efficiency, cache locality, and SIMD hardware acceleration.

## Features

### 🧠 Advanced Tensor Engine
* **Zero-Copy Views:** Support for `reshape`, `transpose`, `slice`, and `broadcast_to` without moving or duplicating underlying memory.
* **SIMD Acceleration:** Core matrix multiplication (`matmul`) is implemented using cache-packed AVX2 and FMA (Fused Multiply-Add) intrinsics for maximum FLOPs/cycle.
* **Multi-threading:** Leverages OpenMP for multi-core parallelism across batch and sequence dimensions.

### 💾 GGUF Memory-Mapped Loading & Quantization
* **Zero-Copy Loader:** Employs `mmap` to load GGUF models directly from disk. Tensors are parsed and bound in milliseconds.
* **Quantized Math:** Supports floating-point decompression and FMA dot products for `Q8_0` and `Q4_0` formats on-the-fly, retaining model accuracy with a fraction of the memory footprint.

### 🧩 Neural Network Architecture (`turbo::nn`)
Turbo features a modular neural network API heavily inspired by PyTorch (`torch.nn.Module`), allowing for clean and composable model architecture:
* **`Linear`**: Dense projections seamlessly integrated with the AVX2 GEMM engine.
* **`RMSNorm`**: Root Mean Square Normalization, replacing standard LayerNorm for modern transformer architectures.
* **`RoPE`**: Rotary Positional Embeddings, optimized with precomputed trigonometric caches for fast inference.
* **`SwiGLU`**: Advanced activation functions used in LLaMA 3 and Mistral.
* **`Embedding`**: Features a highly optimized $O(1)$ zero-copy slice for the **decode phase** (single-token generation) and batched, contiguous memory copies for the **prefill phase**.
* **`MultiHeadAttention`**: Full attention mechanism with causal masking, GQA (Grouped Query Attention) head-repeating logic, and KV cache integration.
* **`TransformerBlock`**: A complete, composable decoder block binding Attention, FFN, and RMSNorm together, ready for the main model loop.

### 🔠 Tokenization & Sampling
* **Iterative BPE Tokenizer:** Fully compliant byte-pair encoding tokenizer that handles word boundary meta-spaces (`▁`) and isolates special tokens before tokenization.
* **Autoregressive Sampler:** Supports Greedy (argmax) and Top-P (nucleus) sampling with adjustable Temperature settings, featuring numerical stability safeguards.

### 🧪 Mathematical Validation
Turbo ships with a robust `GoogleTest` suite. Every single neural network layer is rigorously tested for boundary safety and mathematically validated against deterministic **PyTorch Golden Tests**. This ensures that the custom C++ floating-point operations perfectly match the reference implementations.

## Building

Ensure you have CMake, OpenMP, and a C++17 compliant compiler installed (GCC or Clang).

```bash
# Clone the repository and submodules (if any)
git clone https://github.com/rithvik-2006/Turbo.cpp.git
cd Turbo.cpp

# Create build directory and compile
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Running the Engine

You can run the interactive chat CLI using any compatible `.gguf` model (like TinyLlama 1.1B):

```bash
# Usage: ./turbo_chat <path_to_model.gguf> [temperature]
./build/turbo_chat models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf 0.7
```

## Testing

Run the GoogleTest validation suite to ensure mathematical and memory integrity:
```bash
cd build
ctest --output-on-failure
```
