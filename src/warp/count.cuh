/*
 * Warp-Cooperative Count Operation
 *
 * ALGORITHM:
 *   Similar to search, but counts occurrences instead of returning value.
 *   For a valid hash table, count should be 0 (not found) or 1 (found).
 *
 *   1. Use __ballot_sync to find which lanes want to count
 *   2. For each probe distance (linear probing):
 *      - Each lane reads current slot
 *      - Check status and key, increment count if found
 *      - Stop at EMPTY or after finding key
 *   3. Return count (0 or 1)
 *
 * REFERENCES:
 *   - SlabHash/src/concurrent_map/warp/count.cuh
 *
 * TODO: Implement warp-cooperative count
 */

#pragma once

#include "../hash_map_context.cuh"

template <typename KeyT, typename ValueT>
__device__ __forceinline__ void GpuHashMapContext<KeyT, ValueT>::countKey(
    bool to_count,
    const uint32_t laneId,
    const KeyT& key,
    uint32_t& count,
    uint32_t bucket) {

  // TODO: Implement warp-cooperative count
  // Hints:
  //   - Initialize count = 0
  //   - Similar logic to search
  //   - Set count = 1 if key found
}
