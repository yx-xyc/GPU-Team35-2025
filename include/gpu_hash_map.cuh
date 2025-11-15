/*
 * GPU Hash Map Library
 * Team 35 - GPU Course Project 2025
 *
 * Main API Header
 * This is the only file users need to include to use the hash map library.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <string>

// Error checking macro
#define CHECK_CUDA_ERROR(call)                                              \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,      \
              cudaGetErrorString(err));                                     \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

// Special value indicating key not found
#define SEARCH_NOT_FOUND 0xFFFFFFFF

// Include implementation files
#include "../src/hash_map_context.cuh"
#include "../src/iterator.cuh"
#include "../src/hash_map_impl.cuh"

// Include warp operations
#include "../src/warp/insert.cuh"
#include "../src/warp/search.cuh"
#include "../src/warp/delete.cuh"
#include "../src/warp/count.cuh"

// Include kernels
#include "../src/kernels/build_kernel.cuh"
#include "../src/kernels/search_kernel.cuh"
#include "../src/kernels/delete_kernel.cuh"
#include "../src/kernels/count_kernel.cuh"
