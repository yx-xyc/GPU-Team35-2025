/*
 * Warp-Cooperative Search Operation
 *
 * ALGORITHM:
 *   1. Use __ballot_sync to find which lanes want to search
 *   2. For each probe distance (linear probing):
 *      - Each lane reads current slot = (bucket + probe) % num_buckets
 *      - Check status:
 *        * EMPTY: key doesn't exist, stop searching
 *        * OCCUPIED: compare key, if match found return value
 *        * TOMBSTONE: continue probing
 *      - Use __ballot_sync to find which lanes finished
 *      - Exit when all lanes done or MAX_PROBE_LENGTH reached
 *
 * REFERENCES:
 *   - SlabHash/src/concurrent_map/warp/search.cuh
 *
 * TODO: Implement warp-cooperative search with linear probing
 */

#pragma once

#include "../hash_map_context.cuh"

template <typename KeyT, typename ValueT>
__device__ __forceinline__ void GpuHashMapContext<KeyT, ValueT>::searchKey(
    bool to_search,
    const uint32_t laneId,
    const KeyT& key,
    ValueT& result,
    uint32_t bucket) {

  // TODO: Implement warp-cooperative search
  // Hints:
  //   - Initialize result = SEARCH_NOT_FOUND
  //   - Read d_status_[slot], d_keys_[slot], d_values_[slot]
  //   - Stop at EMPTY slot (key not in table)
  //   - Continue at TOMBSTONE (key might be further along chain)
  //   - Use __threadfence() for memory consistency
}
