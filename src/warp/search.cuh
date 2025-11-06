/*
 * Warp-Cooperative Search Operation
 *
 * ALGORITHM:
 *   1. Use __activemask to get active threads
 *   2. For each probe distance (linear probing):
 *      - Each lane reads current slot = (bucket + probe + laneId) % num_buckets
 *      - Check status:
 *        * EMPTY: key doesn't exist, mark done
 *        * OCCUPIED: compare key, if match found return value and mark done
 *        * TOMBSTONE: continue probing
 *      - Use __any_sync/__all_sync to coordinate exit
 *      - Exit when all lanes done or MAX_PROBE_LENGTH reached
 *
 * IMPORTANT: Uses done-flag pattern instead of early return to maintain warp synchronization
 *
 * REFERENCES:
 *   - SlabHash/src/concurrent_map/warp/search.cuh
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

  // Initialize result to not found
  result = SEARCH_NOT_FOUND;

  // Get active thread mask
  unsigned mask = __activemask();

  // Threads that don't need to search are immediately done
  bool done = !to_search;

  // Linear probing - each thread probes its own sequence
  // Continue until all threads in warp are done OR we hit max probe length
  for (uint32_t probe = 0; probe < MAX_PROBE_LENGTH && __any_sync(mask, !done); probe++) {

    // Only active threads that aren't done do work
    if (!done) {
      // Each thread probes sequentially from its own bucket
      uint32_t slot = (bucket + probe) % num_buckets_;

      // Read status, key, and value from this slot
      uint32_t status = d_status_[slot];
      KeyT slot_key = d_keys_[slot];
      ValueT slot_value = d_values_[slot];

      // Memory fence to ensure reads are complete
      __threadfence();

      // If we found a match, retrieve the value
      if (status == OCCUPIED && slot_key == key) {
        result = slot_value;
        done = true;
      }
      // If we found an EMPTY slot, key doesn't exist
      else if (status == EMPTY) {
        // result remains SEARCH_NOT_FOUND
        done = true;
      }
      // Otherwise continue probing (TOMBSTONE, PENDING, or wrong key)
    }

    // Early exit if all threads are done
    if (__all_sync(mask, done)) {
      break;
    }
  }

  // If loop exits without finding, result remains SEARCH_NOT_FOUND
}
