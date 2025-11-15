/*
 * Delete Kernel - Bulk Delete Operation
 *
 * Launches many threads to delete keys in parallel.
 * Each thread independently searches for and deletes one key.
 *
 * BEHAVIOR:
 *   - Marks deleted slots as TOMBSTONE
 *   - Does not physically remove entries (maintains probing chains)
 *   - Silently succeeds even if key doesn't exist
 *
 * KERNEL PATTERN:
 *   - Similar to build/search kernels
 *   - Each thread: tid = blockIdx.x * blockDim.x + threadIdx.x
 *   - Warps exit early if all threads beyond data range
 *   - Uses same hash function and probing pattern as insert/search
 *
 * REFERENCES:
 *   - SlabHash/src/concurrent_map/device/delete_kernel.cuh
 */

#pragma once
#include "../warp/delete.cuh"
#include "delete_kernel_optimized.cuh"
#include <cuda_runtime.h>

/**
 * Kernel: Delete many keys in parallel
 *
 * Each thread deletes one key using linear probing.
 *
 * Parameters:
 *   ctx - hash map context (contains hash function and table pointers)
 *   input_keys - device array of keys to delete
 *   num_keys - number of keys to delete
 */
template <typename KeyT, typename ValueT>
__global__ void delete_table_kernel(
    GpuHashMapContext<KeyT, ValueT> ctx,
    const KeyT* input_keys,
    uint32_t num_keys) {

  uint32_t tid    = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  // Early exit for warps beyond data range
  if ((tid - laneId) >= num_keys) return;

  bool has_work = (tid < num_keys);
  KeyT key{};
  uint32_t bucket = 0;

  if (has_work) {
    key = input_keys[tid];
    // Use same hash function as insert/search
    bucket = ctx.computeBucket(key);
  }

  // Call warp-cooperative delete
  deleteKey<KeyT, ValueT>(
      ctx.getKeys(), ctx.getStatus(), ctx.getNumBuckets(), has_work, laneId, key, bucket);
}

/**
 * Launch wrapper for delete kernel
 *
 * Parameters:
 *   ctx - hash map context
 *   input_keys - device array of keys to delete
 *   num_keys - number of keys
 *   stream - CUDA stream (optional)
 */
template <typename KeyT, typename ValueT>
inline void launch_delete_kernel(
    GpuHashMapContext<KeyT, ValueT> ctx,
    const KeyT* input_keys,
    uint32_t num_keys,
    cudaStream_t stream = nullptr) {

  if (num_keys == 0) return;

  const uint32_t block = 128;
  const uint32_t grid  = (num_keys + block - 1) / block;

  delete_table_kernel<KeyT, ValueT><<<grid, block, 0, stream>>>(
      ctx, input_keys, num_keys);
}

/**
 * Host-side API: Delete multiple keys from hash table
 *
 * Parameters:
 *   d_keys - device array of keys to delete
 *   num_keys - number of keys
 */
template <typename KeyT, typename ValueT>
void GpuHashMap<KeyT, ValueT>::deleteTable(const KeyT* d_keys, uint32_t num_keys) {
  launch_delete_kernel<KeyT, ValueT>(context_, d_keys, num_keys);
}

/**
 * Host-side API: Delete multiple keys from hash table (OPTIMIZED VERSION)
 *
 * Uses optimized kernel with reduced atomic operations for better performance.
 *
 * Parameters:
 *   d_keys - device array of keys to delete
 *   num_keys - number of keys
 */
template <typename KeyT, typename ValueT>
void GpuHashMap<KeyT, ValueT>::deleteTableOptimized(const KeyT* d_keys, uint32_t num_keys) {
  launch_delete_kernel_optimized<KeyT, ValueT>(context_, d_keys, num_keys);
}
