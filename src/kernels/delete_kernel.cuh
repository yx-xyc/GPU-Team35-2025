/*
 * Delete Kernel - Bulk Delete Operation
 *
 * Launches many threads to delete keys in parallel.
 * Each warp cooperates to delete ~32 keys.
 *
 * BEHAVIOR:
 *   - Marks deleted slots as TOMBSTONE
 *   - Does not physically remove (maintains probing chains)
 *   - Silently succeeds even if key doesn't exist
 *
 * KERNEL PATTERN:
 *   Similar to build/search kernels, but calls deleteKey()
 *
 * REFERENCES:
 *   - SlabHash/src/concurrent_map/device/delete_kernel.cuh
 *
 * TODO: Implement bulk delete kernel and host wrapper
 */

#pragma once

#include "../hash_map_context.cuh"

/*
 * Kernel: Delete many keys in parallel
 *
 * Parameters:
 *   ctx - hash map context
 *   d_keys - device array of keys to delete
 *   num_keys - number of keys
 */
template <typename KeyT, typename ValueT>
__global__ void delete_table_kernel(GpuHashMapContext<KeyT, ValueT> ctx,
                                     const KeyT* d_keys,
                                     uint32_t num_keys) {
  // TODO: Implement kernel
  // Hints:
  //   uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  //   uint32_t laneId = threadIdx.x & 0x1F;
  //
  //   if ((tid - laneId) >= num_keys) return;
  //
  //   bool has_work = (tid < num_keys);
  //   KeyT my_key = has_work ? d_keys[tid] : 0;
  //
  //   uint32_t bucket = ctx.computeBucket(my_key);
  //
  //   ctx.deleteKey(has_work, laneId, my_key, bucket);
}

/*
 * Host wrapper: Launch delete kernel
 */
template <typename KeyT, typename ValueT>
void GpuHashMap<KeyT, ValueT>::deleteTable(const KeyT* d_keys, uint32_t num_keys) {
  // TODO: Implement kernel launcher
  // Similar to buildTable but launches delete_table_kernel
}
