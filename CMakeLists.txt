cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GPUHashMap CUDA CXX)

# Find CUDA
find_package(CUDA 8.0 REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)

# Build type
set(CMAKE_BUILD_TYPE "Release")

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# GPU architecture options
set(GENCODE_SM35
  -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)
set(GENCODE_SM50
  -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50)
set(GENCODE_SM60
  -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60)
set(GENCODE_SM61
  -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61)
set(GENCODE_SM70
  -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70)
set(GENCODE_SM75
  -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75)
set(GENCODE_SM80
  -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80)

# Configure which architectures to build for
option(HASHMAP_GENCODE_SM35 "GENCODE_SM35" OFF)
option(HASHMAP_GENCODE_SM50 "GENCODE_SM50" OFF)
option(HASHMAP_GENCODE_SM60 "GENCODE_SM60" OFF)
option(HASHMAP_GENCODE_SM61 "GENCODE_SM61" OFF)
option(HASHMAP_GENCODE_SM70 "GENCODE_SM70" OFF)
option(HASHMAP_GENCODE_SM75 "GENCODE_SM75" ON)  # Default for RTX 20xx series
option(HASHMAP_GENCODE_SM80 "GENCODE_SM80" OFF)

# Build gencode string
set(GENCODE "")
if(HASHMAP_GENCODE_SM35)
  set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif()
if(HASHMAP_GENCODE_SM50)
  set(GENCODE ${GENCODE} ${GENCODE_SM50})
endif()
if(HASHMAP_GENCODE_SM60)
  set(GENCODE ${GENCODE} ${GENCODE_SM60})
endif()
if(HASHMAP_GENCODE_SM61)
  set(GENCODE ${GENCODE} ${GENCODE_SM61})
endif()
if(HASHMAP_GENCODE_SM70)
  set(GENCODE ${GENCODE} ${GENCODE_SM70})
endif()
if(HASHMAP_GENCODE_SM75)
  set(GENCODE ${GENCODE} ${GENCODE_SM75})
endif()
if(HASHMAP_GENCODE_SM80)
  set(GENCODE ${GENCODE} ${GENCODE_SM80})
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Optional verbose PTXAS
if(CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif()

# Add subdirectories
add_subdirectory(examples)
add_subdirectory(test)
