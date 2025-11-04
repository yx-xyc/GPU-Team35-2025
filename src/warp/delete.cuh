#pragma once
#include "../hash_map_context.cuh"

#ifndef WARP_WIDTH
#define WARP_WIDTH 32
#endif

#ifndef EMPTY
#define EMPTY 0u
#endif
#ifndef TOMBSTONE
#define TOMBSTONE 1u
#endif
#ifndef OCCUPIED
#define OCCUPIED 2u
#endif

// 每个 lane 独立完成的线性探测删除（linear probing + tombstone）
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

  // 循环线性探测，直到所有 lane 都完成
  for (uint32_t probe = 0; __any_sync(mask, !done); probe += WARP_WIDTH) {

    if (!done) {
      uint32_t slot = (bucket + probe + laneId) % num_buckets;
      uint32_t st   = d_status[slot];
      KeyT slot_key = d_keys[slot];

      // 命中：置为 TOMBSTONE
      if (st == OCCUPIED && slot_key == key) {
        atomicCAS(&d_status[slot], OCCUPIED, TOMBSTONE);
        done = true;
      }
      // 空槽：说明 key 不存在
      else if (st == EMPTY) {
        done = true;
      }
    }

    if (__all_sync(mask, done)) break;
  }

/*
 * Warp-Cooperative Delete Operation
 *
 * ALGORITHM:
 *   1. Use __ballot_sync to find which lanes want to delete
 *   2. For each probe distance (linear probing):
 *      - Each lane reads current slot = (bucket + probe) % num_buckets
 *      - Check status:
 *        * EMPTY: key doesn't exist, stop
 *        * OCCUPIED: compare key, if match use atomicCAS to mark TOMBSTONE
 *        * TOMBSTONE: continue probing
 *      - Use __ballot_sync to find which lanes finished
 *      - Exit when all lanes done or MAX_PROBE_LENGTH reached
 *
 * IMPORTANT: Mark as TOMBSTONE, don't actually remove!
 *   - Removing would break linear probing chains
 *   - TOMBSTONE allows probing to continue through deleted slots
 *
 * REFERENCES:
 *   - SlabHash/src/concurrent_map/warp/delete.cuh
 *
 * TODO: Implement warp-cooperative delete with tombstone marking
 */

#pragma once

#include "../hash_map_context.cuh"

template <typename KeyT, typename ValueT>
__device__ __forceinline__ void GpuHashMapContext<KeyT, ValueT>::deleteKey(
    bool to_delete,
    const uint32_t laneId,
    const KeyT& key,
    uint32_t bucket) {

  // TODO: Implement warp-cooperative delete
  // Hints:
  //   - Use atomicCAS(&d_status_[slot], OCCUPIED, TOMBSTONE) to delete
  //   - Don't stop at TOMBSTONE - key might be further in chain
  //   - Stop at EMPTY - key definitely doesn't exist

}
