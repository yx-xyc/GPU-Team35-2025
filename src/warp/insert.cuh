/*
 * Warp-Cooperative Insert Operation
 *
 * ALGORITHM:
 *   1. Use __ballot_sync to find which lanes want to insert
 *   2. For each probe distance (linear probing):
 *      - Each lane tries current slot = (bucket + probe) % num_buckets
 *      - Use atomicCAS to claim EMPTY or TOMBSTONE slots
 *      - Write key and value if successful
 *      - Use __ballot_sync to find which lanes succeeded
 *      - Exit when all lanes done or MAX_PROBE_LENGTH reached
 *
 * REFERENCES:
 *   - SlabHash/src/concurrent_map/warp/insert.cuh (for slab-based version)
 *   - CUDA Programming Guide: Warp Vote Functions (__ballot_sync)
 *
 * TODO: Implement warp-cooperative insert with linear probing
 */

#pragma once

#include "../hash_map_context.cuh"

template <typename KeyT, typename ValueT>
__device__ __forceinline__ void GpuHashMapContext<KeyT, ValueT>::insertKey(
    bool to_insert,
    const uint32_t laneId,
    const KeyT& key,
    const ValueT& value,
    uint32_t bucket) {

  if (!to_insert) return;  // Simple: each thread works independently for now

  // Linear probing
  for (uint32_t probe = 0; probe < MAX_PROBE_LENGTH; probe++) {
    uint32_t slot = (bucket + probe) % num_buckets_;

    // Try to claim EMPTY slot
    uint32_t old_status = atomicCAS(&d_status_[slot], EMPTY, OCCUPIED);

    if (old_status == EMPTY || old_status == TOMBSTONE) {
      // Claimed slot or reusing tombstone - try to claim tombstone too
      if (old_status == TOMBSTONE) {
        old_status = atomicCAS(&d_status_[slot], TOMBSTONE, OCCUPIED);
      }

      if (old_status == EMPTY || old_status == TOMBSTONE) {
        // Successfully claimed - write data
        d_keys_[slot] = key;
        d_values_[slot] = value;
        return;  // Done
      }
    }

    // Check if key already exists at this slot
    if (old_status == OCCUPIED && d_keys_[slot] == key) {
      d_values_[slot] = value;  // Update existing
      return;
    }

    // Collision - continue probing
  }

  // Failed to insert (table full or too many collisions)
}
