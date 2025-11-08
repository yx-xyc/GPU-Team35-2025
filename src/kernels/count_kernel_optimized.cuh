/*
 * Count Kernel - Optimized with Block-level Reduction
 *
 * OPTIMIZATION IDEA:
 *   Instead of every thread doing atomicAdd (causes contention),
 *   we reduce within each block first, then only ONE atomic per block.
 *
 * STEPS:
 *   1. Each thread counts its portion of the table
 *   2. Store counts in shared memory (one per thread)
 *   3. Reduce shared memory values within block
 *   4. Thread 0 adds the block's total to global counter
 */

#pragma once

#include "../hash_map_context.cuh"
#include "../hash_map_impl.cuh"

/*
 * Optimized count kernel using block-level reduction
 */
template <typename KeyT, typename ValueT>
__global__ void count_table_kernel_optimized(
    GpuHashMapContext<KeyT, ValueT> ctx,
    uint32_t* d_count,
    uint32_t num_buckets) {
  
  // Shared memory for this block's thread counts
  __shared__ uint32_t shared_counts[256];  // assuming max 256 threads per block
  
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t local_id = threadIdx.x; // local thread index in block
  uint32_t stride = gridDim.x * blockDim.x; // total threads
  
  // Step 1: Each thread counts occupied slots in its portion
  uint32_t my_count = 0; 
  for (uint32_t i = tid; i < num_buckets; i += stride) {
    uint32_t status = ctx.getStatus()[i];
    if (status == OCCUPIED) {
      my_count++;
    }
  }
  
  // Step 2: Write my count to shared memory
  shared_counts[local_id] = my_count;
  __syncthreads();
  
  // Step 3: Tree-based reduction in shared memory
  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (local_id < s) {
      shared_counts[local_id] += shared_counts[local_id + s];
    }
    __syncthreads();
  }
  
  // Step 4: Thread 0 adds block total to global counter
  if (local_id == 0) {
    atomicAdd(d_count, shared_counts[0]);
  }
}