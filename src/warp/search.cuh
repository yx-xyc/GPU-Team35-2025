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

  // Initialize result to not found
  result = SEARCH_NOT_FOUND;

  // Early exit if this thread doesn't need to search
  if (!to_search) {
    return;
  }

  // Linear probing with warp cooperation
  for (uint32_t probe = 0; probe < MAX_PROBE_LENGTH; probe += WARP_WIDTH) {
    // Calculate slot index for this lane
    uint32_t slot = (bucket + probe + laneId) % num_buckets_;

    // Read status, key, and value from this slot
    uint32_t status = d_status_[slot];
    KeyT slot_key = d_keys_[slot];
    ValueT slot_value = d_values_[slot];

    // Memory fence to ensure reads are complete
    __threadfence();

    // Check conditions across warp
    uint32_t empty_mask = __ballot_sync(0xFFFFFFFF, status == EMPTY);
    uint32_t match_mask = __ballot_sync(0xFFFFFFFF,
        status == OCCUPIED && slot_key == key);

    // If any lane found a match, retrieve the value
    if (match_mask != 0) {
      // Find which lane has the match and broadcast the value
      // If this lane has the match, store the value
      if (status == OCCUPIED && slot_key == key) {
        result = slot_value;
      }
      return;
    }

    // If any lane found EMPTY, key doesn't exist
    if (empty_mask != 0) {
      // result remains SEARCH_NOT_FOUND
      return;
    }

    // Continue probing through TOMBSTONE or wrong keys
  }

  // Exceeded max probe length, key not found
  // result remains SEARCH_NOT_FOUND
}
