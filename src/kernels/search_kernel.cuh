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
 * ONE-WARP-PER-KEY Kernel with Stride Loop Optimization
 *
 * All 32 threads in a warp work together to search for ONE key.
 * They check 32 slots in parallel per iteration.
 *
 * OPTIMIZATION: Uses stride loop so each warp processes MULTIPLE queries
 * (similar to count kernel optimization pattern)
 *
 * ADVANTAGES:
 *   - 32x parallelism per key
 *   - Faster for hard-to-find keys (high load factor)
 *   - Better GPU utilization with stride loop
 *   - Amortizes launch overhead
 *
 * USAGE: Best for small batches (< threshold)
 */
template <typename KeyT, typename ValueT>
__global__ void search_table_one_warp_per_key_kernel(
    GpuHashMapContext<KeyT, ValueT> ctx,
    const KeyT* d_queries,
    ValueT* d_results,
    uint32_t num_queries) {

  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;
  uint32_t warpId = tid >> 5;  // Global warp ID
  uint32_t num_warps = (gridDim.x * blockDim.x) >> 5;  // Total warps

  // STRIDE LOOP: Each warp processes multiple queries
  for (uint32_t query_idx = warpId; query_idx < num_queries; query_idx += num_warps) {

    // ALL 32 threads in this warp work on the SAME query
    KeyT shared_key = d_queries[query_idx];
    ValueT result = SEARCH_NOT_FOUND;

    // Compute bucket (all threads compute same bucket for same key)
    uint32_t bucket = ctx.computeBucket(shared_key);

    // Get active mask
    unsigned mask = __activemask();
    bool done = false;

    // Warp-cooperative search: check 32 slots per iteration
    for (uint32_t probe = 0; probe < GpuHashMapContext<KeyT, ValueT>::MAX_PROBE_LENGTH && __any_sync(mask, !done); probe += 32) {

      if (!done) {
        // Each thread checks a DIFFERENT slot for the SAME key
        uint32_t slot = (bucket + probe + laneId) % ctx.getNumBuckets();

        // Read slot data
        uint32_t status = ctx.getStatus()[slot];
        KeyT slot_key = ctx.getKeys()[slot];
        ValueT slot_value = ctx.getValues()[slot];

        // Memory fence
        __threadfence();

        // Check if this lane found the key
        if (status == OCCUPIED && slot_key == shared_key) {
          result = slot_value;
          done = true;
        }
        // Check if this lane found EMPTY (key doesn't exist)
        else if (status == EMPTY) {
          done = true;
        }
      }

      // Check if any thread found result (success or empty)
      if (__any_sync(mask, done)) {
        break;
      }
    }

    // Broadcast result across warp using shuffle
    uint32_t found_mask = __ballot_sync(mask, result != SEARCH_NOT_FOUND);

    if (found_mask != 0) {
      int src_lane = __ffs(found_mask) - 1;
      result = __shfl_sync(mask, result, src_lane);
    }

    // Only lane 0 writes the result
    if (laneId == 0) {
      d_results[query_idx] = result;
    }
  }
}

/*
 * Host wrapper: Launch search kernel with hybrid strategy
 *
 * Automatically selects between two approaches based on batch size:
 *   - Small batches (< threshold): One-warp-per-key (32x parallelism per key)
 *   - Large batches (>= threshold): One-thread-per-key (higher throughput)
 */
template <typename KeyT, typename ValueT>
void GpuHashMap<KeyT, ValueT>::searchTable(const KeyT* d_queries,
                                            ValueT* d_results,
                                            uint32_t num_queries) {
  CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

  const uint32_t block_size = 128;

  // Hybrid strategy: choose kernel based on batch size
  if (num_queries < search_warp_threshold_) {
    // Small batch: Use one-warp-per-key (32x parallelism per key)
    // Need num_queries warps, each block has block_size/32 warps
    const uint32_t warps_per_block = block_size / 32;
    const uint32_t num_blocks = (num_queries + warps_per_block - 1) / warps_per_block;

    if (verbose_) {
      std::cout << "[GpuHashMap] Search: " << num_queries
                << " queries, using one-warp-per-key kernel (threshold: "
                << search_warp_threshold_ << ")" << std::endl;
    }

    search_table_one_warp_per_key_kernel<<<num_blocks, block_size>>>(
        context_,
        d_queries,
        d_results,
        num_queries);
  } else {
    // Large batch: Use one-thread-per-key (higher throughput)
    const uint32_t num_blocks = (num_queries + block_size - 1) / block_size;

    if (verbose_) {
      std::cout << "[GpuHashMap] Search: " << num_queries
                << " queries, using one-thread-per-key kernel (threshold: "
                << search_warp_threshold_ << ")" << std::endl;
    }

    search_table_kernel<<<num_blocks, block_size>>>(
        context_,
        d_queries,
        d_results,
        num_queries);
  }

  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

/*
 * Force warp-per-key strategy (ignores threshold)
 *
 * Directly calls one-warp-per-key kernel with stride loop optimization.
 * Uses fixed grid size for better GPU utilization.
 */
template <typename KeyT, typename ValueT>
void GpuHashMap<KeyT, ValueT>::searchTableWarpPerKey(const KeyT* d_queries,
                                                      ValueT* d_results,
                                                      uint32_t num_queries) {
  CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

  const uint32_t block_size = 128;
  // Use fixed grid size for stride loop (better utilization)
  const uint32_t num_blocks = 256;

  if (verbose_) {
    std::cout << "[GpuHashMap] Search (WARP-PER-KEY with stride): " << num_queries
              << " queries" << std::endl;
  }

  search_table_one_warp_per_key_kernel<<<num_blocks, block_size>>>(
      context_,
      d_queries,
      d_results,
      num_queries);

  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

/*
 * Force thread-per-key strategy (ignores threshold)
 *
 * Directly calls one-thread-per-key kernel.
 * Each thread searches its own key independently.
 */
template <typename KeyT, typename ValueT>
void GpuHashMap<KeyT, ValueT>::searchTableThreadPerKey(const KeyT* d_queries,
                                                        ValueT* d_results,
                                                        uint32_t num_queries) {
  CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

  const uint32_t block_size = 128;
  const uint32_t num_blocks = (num_queries + block_size - 1) / block_size;

  if (verbose_) {
    std::cout << "[GpuHashMap] Search (FORCED THREAD-PER-KEY): " << num_queries
              << " queries" << std::endl;
  }

  search_table_kernel<<<num_blocks, block_size>>>(
      context_,
      d_queries,
      d_results,
      num_queries);

  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

