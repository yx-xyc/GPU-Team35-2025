/*
 * Warp-Cooperative Delete Operation
 *
 * Each thread in a warp independently searches for and deletes its own key
 * using linear probing with TOMBSTONE marking.
 *
 * ALGORITHM:
 *   1. Each thread probes sequentially from its bucket: (bucket + probe) % num_buckets
 *   2. For each slot, check status:
 *      - EMPTY: key doesn't exist, stop
 *      - OCCUPIED: compare key, if match use atomicCAS to mark TOMBSTONE
 *      - TOMBSTONE: continue probing (key might be further in chain)
 *   3. Exit when done or MAX_PROBE_LENGTH reached
 *
 * IMPORTANT: Mark as TOMBSTONE, don't actually remove!
 *   - Removing would break linear probing chains
 *   - TOMBSTONE allows probing to continue through deleted slots
 *
 * REFERENCES:
 *   - SlabHash/src/concurrent_map/warp/delete.cuh
 */

#pragma once
#include "../hash_map_context.cuh"

/**
 * Delete a key from the hash table using linear probing
 *
 * This function is called by the delete kernel. Each thread searches for its key
 * independently using the same probing pattern as insert/search.
 *
 * Parameters:
 *   d_keys - device array of keys
 *   d_status - device array of slot statuses
 *   num_buckets - total number of buckets
 *   to_delete - true if this thread has a key to delete
 *   laneId - lane ID within warp (0-31, unused but kept for consistency)
 *   key - key to delete
 *   bucket - starting bucket computed from hash function
 */
template <typename KeyT, typename ValueT>
__device__ __forceinline__ void deleteKey(
    KeyT* d_keys,
    uint32_t* d_status,
    uint32_t num_buckets,
    bool to_delete,
    uint32_t laneId,
    const KeyT& key,
    uint32_t bucket) {

  unsigned mask = __activemask();
  bool done = !to_delete;

  // Linear probing with same pattern as insert/search
  const uint32_t MAX_PROBE = 128; // Same as GpuHashMapContext::MAX_PROBE_LENGTH
  for (uint32_t probe = 0; probe < MAX_PROBE && __any_sync(mask, !done); probe++) {

    if (!done) {
      // Probe sequentially: same pattern as insert/search
      uint32_t slot = (bucket + probe) % num_buckets;
      uint32_t st   = d_status[slot];
      KeyT slot_key = d_keys[slot];

      // Memory fence to ensure reads are complete
      __threadfence();

      // If key found, mark as TOMBSTONE
      if (st == OCCUPIED && slot_key == key) {
        uint32_t old = atomicCAS(&d_status[slot], OCCUPIED, TOMBSTONE);
        if (old == OCCUPIED) {
          // Successfully deleted
          __threadfence(); // Ensure delete is visible to other threads
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

    if (__all_sync(mask, done)) break;
  }
}
