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

  // Debug: track keys 1-5 only
  bool debug = (key >= 1 && key <= 5);

  // Linear probing with three-state protocol
  for (uint32_t probe = 0; probe < MAX_PROBE_LENGTH; probe++) {
    uint32_t slot = (bucket + probe) % num_buckets_;

    // READ FIRST: Check current slot status
    uint32_t slot_status = d_status_[slot];

    // If OCCUPIED, check for duplicate key
    if (slot_status == OCCUPIED) {
      __threadfence();  // Ensure key write is visible
      KeyT slot_key = d_keys_[slot];
      if (debug) printf("[INSERT] tid=%d key=%u CHECK OCCUPIED slot=%u: slot_key=%u, match=%d\n",
                        threadIdx.x + blockIdx.x * blockDim.x, key, slot, slot_key, (slot_key == key));
      if (slot_key == key) {
        // Duplicate found - update value
        if (debug) printf("[INSERT] tid=%d key=%u DUPLICATE at slot=%u, updating\n",
                          threadIdx.x + blockIdx.x * blockDim.x, key, slot);
        d_values_[slot] = value;
        return;
      }
      // Different key - collision, continue probing
      if (debug) printf("[INSERT] tid=%d key=%u COLLISION at slot=%u (has key=%u)\n",
                        threadIdx.x + blockIdx.x * blockDim.x, key, slot, slot_key);
      continue;
    }

    // If PENDING, another thread is writing - retry this slot
    if (slot_status == PENDING) {
      if (debug) printf("[INSERT] tid=%d key=%u PENDING at slot=%u, retrying\n",
                        threadIdx.x + blockIdx.x * blockDim.x, key, slot);
      probe--;  // Retry same slot in next iteration
      continue;
    }

    // Try to claim EMPTY slot with PENDING state
    if (slot_status == EMPTY) {
      uint32_t old = atomicCAS(&d_status_[slot], EMPTY, PENDING);
      if (old == EMPTY) {
        // Successfully claimed - write key/value, then mark OCCUPIED
        if (debug) printf("[INSERT] tid=%d key=%u CLAIMED EMPTY slot=%u (bucket=%u probe=%u)\n",
                          threadIdx.x + blockIdx.x * blockDim.x, key, slot, bucket, probe);
        d_keys_[slot] = key;
        d_values_[slot] = value;
        __threadfence();  // Ensure writes are visible
        d_status_[slot] = OCCUPIED;  // Mark as ready
        if (debug) printf("[INSERT] tid=%d key=%u COMPLETED at slot=%u\n",
                          threadIdx.x + blockIdx.x * blockDim.x, key, slot);
        return;
      }
      // CAS failed - retry this slot
      probe--;
      continue;
    }

    // Try to reuse TOMBSTONE slot with PENDING state
    if (slot_status == TOMBSTONE) {
      uint32_t old = atomicCAS(&d_status_[slot], TOMBSTONE, PENDING);
      if (old == TOMBSTONE) {
        // Successfully claimed tombstone - write key/value, then mark OCCUPIED
        if (debug) printf("[INSERT] tid=%d key=%u CLAIMED TOMBSTONE slot=%u\n",
                          threadIdx.x + blockIdx.x * blockDim.x, key, slot);
        d_keys_[slot] = key;
        d_values_[slot] = value;
        __threadfence();  // Ensure writes are visible
        d_status_[slot] = OCCUPIED;  // Mark as ready
        return;
      }
      // CAS failed - retry this slot
      probe--;
      continue;
    }

    // Unexpected status - continue probing
  }

  // Failed to insert (table full or too many collisions)
  if (debug) printf("[INSERT] tid=%d key=%u FAILED after MAX_PROBE_LENGTH\n",
                    threadIdx.x + blockIdx.x * blockDim.x, key);
}
