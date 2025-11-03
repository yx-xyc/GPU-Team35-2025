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

  // Initialize count to 0
  count = 0;

  // Early exit if this thread doesn't need to count
  if (!to_count) {
    return;
  }

  // Linear probing with warp cooperation
  for (uint32_t probe = 0; probe < MAX_PROBE_LENGTH; probe += WARP_WIDTH) {
    // Calculate slot index for this lane
    uint32_t slot = (bucket + probe + laneId) % num_buckets_;

    // Read status and key from this slot
    uint32_t status = d_status_[slot];
    KeyT slot_key = d_keys_[slot];

    // Memory fence to ensure reads are complete
    __threadfence();

    // Check conditions across warp
    uint32_t empty_mask = __ballot_sync(0xFFFFFFFF, status == EMPTY);
    uint32_t match_mask = __ballot_sync(0xFFFFFFFF, 
        status == OCCUPIED && slot_key == key);

    // If any lane found a match, count it
    if (match_mask != 0) {
      // Count number of matches (should be 1 for valid hash table)
      count = __popc(match_mask);
      return;
    }

    // If any lane found EMPTY, key doesn't exist
    if (empty_mask != 0) {
      // count remains 0
      return;
    }

    // Continue probing through TOMBSTONE or wrong keys
  }

  // Exceeded max probe length, key not found
  // count remains 0
}
