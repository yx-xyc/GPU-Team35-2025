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

  // Initialize count to 0
  count = 0;

  // Get active thread mask
  unsigned mask = __activemask();

  // Threads that don't need to count are immediately done
  bool done = !to_count;

  // Linear probing - each thread probes its own sequence
  for (uint32_t probe = 0; probe < MAX_PROBE_LENGTH && __any_sync(mask, !done); probe ++) {

    // Only active threads that aren't done do work
    if (!done) {
      // Each thread probes sequentially from its own bucket
      uint32_t slot = (bucket + probe) % num_buckets_;

      // Read status and key from this slot
      uint32_t status = d_status_[slot];
      KeyT slot_key = d_keys_[slot];

      // Memory fence to ensure reads are complete
      __threadfence();

      // If we found a match, count it
      if (status == OCCUPIED && slot_key == key) {
        count = 1;  // This thread found its key
        done = true;
      }
      // If we found an EMPTY slot, key doesn't exist
      else if (status == EMPTY) {
        // count remains 0
        done = true;
      }
      // Otherwise continue probing (TOMBSTONE or wrong key)
    }

    // Early exit if all threads are done
    if (__all_sync(mask, done)) {
      break;
    }
  }

  // If loop exits without finding, count remains 0
}
