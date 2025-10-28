/*
 * Warp-Cooperative Delete Operation
 *
 * ALGORITHM:
 *   1. Use __ballot_sync to find which lanes want to delete
 *   2. For each probe distance (linear probing):
 *      - Each lane reads current slot = (bucket + probe) % num_buckets
 *      - Check status:
 *        * EMPTY: key doesn't exist, stop
 *        * OCCUPIED: compare key, if match use atomicCAS to mark TOMBSTONE
 *        * TOMBSTONE: continue probing
 *      - Use __ballot_sync to find which lanes finished
 *      - Exit when all lanes done or MAX_PROBE_LENGTH reached
 *
 * IMPORTANT: Mark as TOMBSTONE, don't actually remove!
 *   - Removing would break linear probing chains
 *   - TOMBSTONE allows probing to continue through deleted slots
 *
 * REFERENCES:
 *   - SlabHash/src/concurrent_map/warp/delete.cuh
 *
 * TODO: Implement warp-cooperative delete with tombstone marking
 */

#pragma once

#include "../hash_map_context.cuh"

template <typename KeyT, typename ValueT>
__device__ __forceinline__ void GpuHashMapContext<KeyT, ValueT>::deleteKey(
    bool to_delete,
    const uint32_t laneId,
    const KeyT& key,
    uint32_t bucket) {

  // TODO: Implement warp-cooperative delete
  // Hints:
  //   - Use atomicCAS(&d_status_[slot], OCCUPIED, TOMBSTONE) to delete
  //   - Don't stop at TOMBSTONE - key might be further in chain
  //   - Stop at EMPTY - key definitely doesn't exist
}
