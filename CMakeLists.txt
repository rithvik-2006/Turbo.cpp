cmake_minimum_required(VERSION 3.14)
project(TurboEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Unlock AVX2 and FMA hardware intrinsics, OpenMP, and -O3 optimization
add_compile_options(-mavx2 -mfma -fopenmp -O3)
add_link_options(-fopenmp)

add_executable(benchmark_gemm benchmarks/benchmark_gemm.cpp)
target_link_libraries(benchmark_gemm PRIVATE tensor_lib OpenMP::OpenMP_CXX)
target_compile_options(benchmark_gemm PRIVATE -O3 -mavx2 -mfma)

include_directories(include)

add_library(tensor_lib src/tensor/tensor.cpp src/nn/softmax.cpp src/nn/linear.cpp src/nn/swiglu.cpp src/nn/embedding.cpp src/nn/rmsnorm.cpp src/nn/rope.cpp src/nn/attention.cpp src/nn/ffn.cpp src/nn/transforrmer_block.cpp src/model/minigpt.cpp src/core/sampler.cpp src/core/generate.cpp)
target_include_directories(tensor_lib PUBLIC include)

find_package(OpenMP REQUIRED)
target_link_libraries(tensor_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(turbo src/turbo.cpp)
target_link_libraries(turbo PRIVATE tensor_lib)

add_executable(run_generation examples/run_generation.cpp)
target_link_libraries(run_generation PRIVATE tensor_lib OpenMP::OpenMP_CXX)
target_compile_options(run_generation PRIVATE -O3 -mavx2 -mfma)

# Add the loader source to your core library
target_sources(tensor_lib PRIVATE src/loader/gguf_loader.cpp)

# Create the test executable
add_executable(run_loader examples/run_loader.cpp)

# Link the executable to your core library
target_link_libraries(run_loader PRIVATE tensor_lib)

enable_testing()
add_subdirectory(tests)