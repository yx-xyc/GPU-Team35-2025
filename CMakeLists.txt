cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Set CUDA architectures (for RTX 2080 Ti) - MUST be before project()
set(CMAKE_CUDA_ARCHITECTURES 75)

project(GPUHashMap LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# Build type
set(CMAKE_BUILD_TYPE "Release")

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Note: GPU architecture now set via CMAKE_CUDA_ARCHITECTURES above
# To change: set CMAKE_CUDA_ARCHITECTURES to your GPU's compute capability
# Examples: 75 (RTX 2080), 80 (A100), 86 (RTX 3090)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Optional verbose PTXAS
option(CUDA_VERBOSE_PTXAS "Enable verbose PTXAS output" OFF)
if(CUDA_VERBOSE_PTXAS)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")
endif()

# Add subdirectories
add_subdirectory(examples)
add_subdirectory(test)
