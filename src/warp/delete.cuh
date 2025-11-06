#pragma once
#include "../hash_map_context.cuh"

#ifndef WARP_WIDTH
#define WARP_WIDTH 32
#endif


#ifndef TOMBSTONE
#define TOMBSTONE 1u

#ifndef EMPTY
#define EMPTY 0u

#endif
#ifndef OCCUPIED
#define OCCUPIED 1u
#endif
#ifndef TOMBSTONE
#define TOMBSTONE 2u
#endif

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

  
  for (uint32_t probe = 0; __any_sync(mask, !done); probe += WARP_WIDTH) {

    if (!done) {
      uint32_t slot = (bucket + probe + laneId) % num_buckets;
      uint32_t st   = d_status[slot];
      KeyT slot_key = d_keys[slot];

      if (st == OCCUPIED && slot_key == key) {
        atomicCAS(&d_status[slot], OCCUPIED, TOMBSTONE);
        done = true;
      }
  
      else if (st == EMPTY) {
        done = true;
      }
    }

    if (__all_sync(mask, done)) break;
  }
}


#pragma once

#include "../hash_map_context.cuh"

template <typename KeyT, typename ValueT>

template <typename KeyT, typename ValueT>
__device__ __forceinline__ void GpuHashMapContext<KeyT, ValueT>::deleteKey(
    bool to_delete,
    const uint32_t laneId,
    const KeyT& key,
    uint32_t bucket) {

  // Minimal-correct warp-cooperative delete:
  // We let lane 0 in each warp do scalar probing to ensure correctness.
  // Other lanes return immediately. This preserves API without touching count kernel.
  if (!to_delete) return;
  if (laneId != 0) return;

  uint32_t pos = bucket;
  for (uint32_t step = 0; step < num_buckets_; ++step) {
    uint32_t st = d_status_[pos];
    if (st == EMPTY) {
      // Reached empty slot: key not present
      return;
    }
    if (st == OCCUPIED && d_keys_[pos] == key) {
      // Mark as tombstone so probing chains stay intact.
      __threadfence();
      atomicExch(&d_status_[pos], TOMBSTONE);
      return;
    }
    pos = (pos + 1) % num_buckets_;
  }
}

