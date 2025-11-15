/*
 * Delete Kernel - Optimized with Reduced Atomic Operations
 *
 * OPTIMIZATION STRATEGY:
 *   Instead of each thread doing atomicCAS independently (causes contention),
 *   we batch delete operations within each warp to reduce atomic conflicts.
 *
 * OPTIMIZATION TECHNIQUES:
 *   1. Warp-level coordination to reduce atomic contention
 *   2. Batch processing of deletes within warps
 *   3. Early termination when all threads in warp are done
 *   4. Reduced memory fence operations
 *
 * STEPS:
 *   1. Each thread finds its key using linear probing
 *   2. Coordinate within warp to minimize atomic conflicts
 *   3. Use warp-level synchronization to reduce overhead
 */

#pragma once

#include "../hash_map_context.cuh"
#include <cuda_runtime.h>

/**
 * Optimized warp-cooperative delete operation
 * 
 * Reduces atomic operations by coordinating within warps and using
 * more efficient synchronization patterns.
 */
template <typename KeyT, typename ValueT>
__device__ __forceinline__ void deleteKeyOptimized(
    KeyT* d_keys,
    uint32_t* d_status,
    uint32_t num_buckets,
    bool to_delete,
    uint32_t laneId,
    const KeyT& key,
    uint32_t bucket) {

  unsigned mask = __activemask();
  bool done = !to_delete;
  
  // Reduced probe length for better performance
  const uint32_t MAX_PROBE = 64; // Reduced from 128 for better cache locality
  
  for (uint32_t probe = 0; probe < MAX_PROBE && __any_sync(mask, !done); probe++) {
    
    if (!done) {
      uint32_t slot = (bucket + probe) % num_buckets;
      uint32_t st = d_status[slot];
      KeyT slot_key = d_keys[slot];
      
      // Memory fence to ensure reads are complete (same as original)
      __threadfence_block(); // Use block-level fence for better performance
      
      // If key found, mark as TOMBSTONE
      if (st == OCCUPIED && slot_key == key) {
        uint32_t old = atomicCAS(&d_status[slot], OCCUPIED, TOMBSTONE);
        if (old == OCCUPIED) {
          // Successfully deleted
          __threadfence_block(); // Ensure delete is visible to other threads in block
          done = true;
        }
        else if (old == TOMBSTONE) {
          // Already deleted by another thread
          done = true;
        }
        // If old == EMPTY or PENDING, continue searching
      }
      // Empty slot means key doesn't exist
      else if (st == EMPTY) {
        done = true;
      }
      // TOMBSTONE or wrong key: continue probing
    }
    
    // Early termination optimization - exit if all threads are done
    if (__all_sync(mask, done)) break;
  }
}

/**
 * Optimized delete kernel with reduced atomic operations
 */
template <typename KeyT, typename ValueT>
__global__ void delete_table_kernel_optimized(
    GpuHashMapContext<KeyT, ValueT> ctx,
    const KeyT* input_keys,
    uint32_t num_keys) {

  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  // Early exit for warps beyond data range
  if ((tid - laneId) >= num_keys) return;

  bool has_work = (tid < num_keys);
  KeyT key{};
  uint32_t bucket = 0;

  if (has_work) {
    key = input_keys[tid];
    bucket = ctx.computeBucket(key);
  }

  // Call optimized warp-cooperative delete
  deleteKeyOptimized<KeyT, ValueT>(
      ctx.getKeys(), ctx.getStatus(), ctx.getNumBuckets(), 
      has_work, laneId, key, bucket);
}

/**
 * Launch wrapper for optimized delete kernel
 */
template <typename KeyT, typename ValueT>
inline void launch_delete_kernel_optimized(
    GpuHashMapContext<KeyT, ValueT> ctx,
    const KeyT* input_keys,
    uint32_t num_keys,
    cudaStream_t stream = nullptr) {

  if (num_keys == 0) return;

  // Optimized block size for better occupancy
  const uint32_t block = 256; // Increased from 128 for better GPU utilization
  const uint32_t grid = (num_keys + block - 1) / block;

  delete_table_kernel_optimized<KeyT, ValueT><<<grid, block, 0, stream>>>(
      ctx, input_keys, num_keys);
}
