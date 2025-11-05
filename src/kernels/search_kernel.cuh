/*
 * Search Kernel - Bulk Search Operation
 *
 * Launches many threads to search for keys in parallel.
 * Each warp cooperates to search ~32 keys.
 *
 * OUTPUT:
 *   - d_results[i] = value if key found
 *   - d_results[i] = SEARCH_NOT_FOUND if key not found
 *
 * KERNEL PATTERN:
 *   Similar to build kernel, but calls searchKey() instead of insertKey()
 *
 * REFERENCES:
 *   - SlabHash/src/concurrent_map/device/search_kernel.cuh
 *
 * TODO: Implement bulk search kernel and host wrapper
 */

#pragma once

#include "../hash_map_context.cuh"

/*
 * Kernel: Search for many keys in parallel
 *
 * Parameters:
 *   ctx - hash map context
 *   d_queries - device array of query keys
 *   d_results - device array for results (output)
 *   num_queries - number of queries
 */
template <typename KeyT, typename ValueT>
__global__ void search_table_kernel(GpuHashMapContext<KeyT, ValueT> ctx,
                                     const KeyT* d_queries,
                                     ValueT* d_results,
                                     uint32_t num_queries) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_queries) return;

  bool has_work = (tid < num_queries);
  KeyT my_query = has_work ? d_queries[tid] : 0;
  ValueT my_result = SEARCH_NOT_FOUND;

  uint32_t bucket = ctx.computeBucket(my_query);

  ctx.searchKey(has_work, laneId, my_query, my_result, bucket);

  if (has_work) {
    d_results[tid] = my_result;
  }
}

/*
 * Host wrapper: Launch search kernel
 */
template <typename KeyT, typename ValueT>
void GpuHashMap<KeyT, ValueT>::searchTable(const KeyT* d_queries,
                                            ValueT* d_results,
                                            uint32_t num_queries) {
  const uint32_t block_size = 128;
  const uint32_t num_blocks = (num_queries + block_size - 1) / block_size;

  CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

  search_table_kernel<<<num_blocks, block_size>>>(
      context_,
      d_queries,
      d_results,
      num_queries);

  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
