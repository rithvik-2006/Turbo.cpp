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

add_library(tensor_lib src/tensor/tensor.cpp)
target_include_directories(tensor_lib PUBLIC include)

find_package(OpenMP REQUIRED)
target_link_libraries(tensor_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(turbo src/turbo.cpp)
target_link_libraries(turbo PRIVATE tensor_lib)

enable_testing()
add_subdirectory(tests)