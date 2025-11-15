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
 */

#pragma once
#include <stdio.h>
#include "../hash_map_context.cuh"

template <typename KeyT, typename ValueT>
__device__ __forceinline__ void GpuHashMapContext<KeyT, ValueT>::insertKey(
    bool to_insert,
    const uint32_t laneId,
    const KeyT& key,
    const ValueT& value,
    uint32_t bucket) {

  if (!to_insert) return;

  // Linear probing with three-state protocol
  for (uint32_t probe = 0; probe < MAX_PROBE_LENGTH; probe++) {
    uint32_t slot = (bucket + probe) % num_buckets_;

    // READ FIRST: Check current slot status
    uint32_t slot_status = d_status_[slot];

    // If OCCUPIED, check for duplicate key
    if (slot_status == OCCUPIED) {
      __threadfence();  // Ensure key write is visible
      KeyT slot_key = d_keys_[slot];
      if (slot_key == key) {
        // Duplicate found - update value
        d_values_[slot] = value;
        return;
      }
      // Different key - collision, continue probing
      continue;
    }

    // If PENDING, another thread is writing - wait briefly, then check for duplicate
    if (slot_status == PENDING) {
      // Bounded busy-wait for PENDING to resolve to OCCUPIED
      const int MAX_WAIT_ITERATIONS = 100000000;
      int wait_iter = 0;
      uint32_t current_status = PENDING;

      while (wait_iter < MAX_WAIT_ITERATIONS && current_status == PENDING) {
        __threadfence();  // Force memory visibility
        current_status = d_status_[slot];
        wait_iter++;
      }

      // After wait, check what happened
      if (current_status == OCCUPIED) {
        // PENDING resolved to OCCUPIED - check for duplicate
        __threadfence();
        KeyT slot_key = d_keys_[slot];
        if (slot_key == key) {
          // Duplicate found
          d_values_[slot] = value;
          return;
        }
        // Different key - collision, continue probing
        continue;
      } else if (current_status == PENDING) {
        // Timeout - still PENDING, skip to avoid infinite loop
        // WARNING: This may allow duplicate insertions!
        printf("[WARNING] PENDING timeout at slot=%u after %d iterations - potential duplicate insertion! (tid=%d, key=%u)\n",
               slot, MAX_WAIT_ITERATIONS, threadIdx.x + blockIdx.x * blockDim.x, key);
        continue;
      } else {
        // Became something else (EMPTY/TOMBSTONE) - recheck this slot
        probe--;  // Recheck this slot
        continue;
      }
    }

    // Try to claim EMPTY or TOMBSTONE slot with PENDING state
    if (slot_status == EMPTY || slot_status == TOMBSTONE) {
      uint32_t old = atomicCAS(&d_status_[slot], slot_status, PENDING);
      if (old == slot_status) {
        // Successfully claimed - write key/value, then mark OCCUPIED
        d_keys_[slot] = key;
        d_values_[slot] = value;
        __threadfence();  // Ensure writes are visible
        d_status_[slot] = OCCUPIED;  // Mark as ready
        return;
      }
      // CAS failed - check what the slot became
      if (old == PENDING) {
        // Another thread just claimed it - need to wait and check for duplicate
        probe--;  // Recheck this slot (will hit PENDING handler above)
        continue;
      }
      // Became something else - recheck
      probe--;
      continue;
    }
  
    printf("[ERROR] insertKey: Unexpected slot status %u at slot=%u (tid=%d, key=%u)\n",
           slot_status, slot, threadIdx.x + blockIdx.x * blockDim.x, key);
    // Unexpected status - continue probing
  }

  // Failed to insert (table full or too many collisions)
}
