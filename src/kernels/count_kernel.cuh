/*
 * Count Kernel - Count Total Elements
 *
 * Counts total number of valid (OCCUPIED) entries in hash table.
 *
 * ALGORITHM:
 *   1. Each thread counts a portion of the table
 *   2. Use atomicAdd to accumulate counts
 *   OR
 *   1. Each thread counts locally
 *   2. Use parallel reduction to sum
 *
 * SIMPLER APPROACH (for starter):
 *   Launch kernel with many threads, each checks one slot.
 *   Use atomicAdd to increment global counter.
 *
 * REFERENCES:
 *   - CUDA SDK samples: reduction
 *   - SlabHash/src/concurrent_map/device/count_kernel.cuh
 */

#pragma once

#include "../hash_map_context.cuh"
#include "../hash_map_impl.cuh"
#include "count_kernel_optimized.cuh"

/*
 * Kernel: Count occupied slots
 *
 * Parameters:
 *   ctx - hash map context
 *   d_count - device pointer to output count (single uint32_t)
 *   num_buckets - total number of buckets to check
 */
template <typename KeyT, typename ValueT>
__global__ void count_table_kernel(GpuHashMapContext<KeyT, ValueT> ctx,
                                    uint32_t* d_count,
                                    uint32_t num_buckets) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;

  for (uint32_t i = tid; i < num_buckets; i += stride) {
    // check state of slot
    uint32_t status = ctx.getStatus()[i];
    
    // if occupied, increment count
    if (status == OCCUPIED) {
      atomicAdd(d_count, 1);
    }
  }
}

/*
 * Host wrapper: Count total elements
 *
 * Returns:
 *   Number of OCCUPIED slots in table
 */
template <typename KeyT, typename ValueT>
uint32_t GpuHashMap<KeyT, ValueT>::countTable() {
  
  uint32_t* d_count;
  CHECK_CUDA_ERROR(cudaMalloc(&d_count, sizeof(uint32_t)));

  CHECK_CUDA_ERROR(cudaMemset(d_count, 0, sizeof(uint32_t)));

  const uint32_t block_size = 256;
  
  uint32_t grid_size = (num_buckets_ + block_size - 1) / block_size;
  grid_size = min(grid_size, 2048u); // limitation of max grid size

  // start kernel to count occupied slots
  count_table_kernel<<<grid_size, block_size>>>(
      context_, 
      d_count,  
      num_buckets_
  );

  CHECK_CUDA_ERROR(cudaGetLastError());
  
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  uint32_t h_count = 0;

  CHECK_CUDA_ERROR(cudaMemcpy(&h_count, 
                              d_count, 
                              sizeof(uint32_t), 
                              cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_count));

  return h_count;
}

/*
 * Host wrapper: Count total elements (OPTIMIZED VERSION)
 */
template <typename KeyT, typename ValueT>
uint32_t GpuHashMap<KeyT, ValueT>::countTableOptimized() {
  
  uint32_t* d_count;
  CHECK_CUDA_ERROR(cudaMalloc(&d_count, sizeof(uint32_t)));
  CHECK_CUDA_ERROR(cudaMemset(d_count, 0, sizeof(uint32_t)));

  const uint32_t block_size = 256;
  uint32_t grid_size = (num_buckets_ + block_size - 1) / block_size;
  grid_size = min(grid_size, 2048u);

  count_table_kernel_optimized<<<grid_size, block_size>>>(
      context_, 
      d_count,  
      num_buckets_
  );

  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  uint32_t h_count = 0;
  CHECK_CUDA_ERROR(cudaMemcpy(&h_count, 
                              d_count, 
                              sizeof(uint32_t), 
                              cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_count));

  return h_count;
}