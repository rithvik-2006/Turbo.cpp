cmake_minimum_required(VERSION 3.14)
project(TurboEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(tensor_lib src/tensor.cpp)
target_include_directories(tensor_lib PUBLIC include)

add_executable(turbo src/turbo.cpp)
target_link_libraries(turbo PRIVATE tensor_lib)

enable_testing()
add_subdirectory(tests)