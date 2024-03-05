cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(driss_torch LANGUAGES CXX CUDA)

# Set the C++ standard for all targets
set(CMAKE_CXX_STANDARD 20) # This might be unsafe since pytorch use std17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable better clangd support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Python3 REQUIRED COMPONENTS Development)
execute_process(
        COMMAND "${Python3_EXECUTABLE}" "-c" "import torch;print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE PT_CMAKE_PREFIX
        COMMAND_ECHO STDOUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
)

# cache CUDA_ARCHITECTURES, which seems to be reset by Torch
set(TMP_STORE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${PT_CMAKE_PREFIX})

find_package(Torch REQUIRED CONFIG)

#  simple_cuda source files
file(GLOB_RECURSE SIMPLE_CUDA_CU_SOURCES src/*.cu)
file(GLOB_RECURSE SIMPLE_CUDA_CPP_SOURCES src/*.cpp)
MESSAGE(STATUS "SIMPLE_CUDA_CU_SOURCES: ${SIMPLE_CUDA_CU_SOURCES}")
MESSAGE(STATUS "SIMPLE_CUDA_CPP_SOURCES: ${SIMPLE_CUDA_CPP_SOURCES}")

add_library(driss_torch SHARED
    ${SIMPLE_CUDA_CU_SOURCES}
    ${SIMPLE_CUDA_CPP_SOURCES}
)
# Add include directories to the library
target_include_directories(driss_torch PUBLIC src/include)

# target_compile_features(cuda_torch PRIVATE cxx_std_11)
target_link_libraries(driss_torch PRIVATE ${TORCH_LIBRARIES} Python3::Python)
