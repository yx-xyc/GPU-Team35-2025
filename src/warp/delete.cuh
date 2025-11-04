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
}
