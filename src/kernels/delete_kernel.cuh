
#pragma once
#include "../warp/delete.cuh"
#include <cuda_runtime.h>

template <typename KeyT, typename ValueT>
__global__ void delete_table_kernel(
    KeyT* d_keys,
    uint32_t* d_status,
    uint32_t num_buckets,
    const KeyT* input_keys,
    uint32_t num_keys) {

  const uint32_t tid    = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t laneId = threadIdx.x & 0x1F;
  if ((tid - laneId) >= num_keys) return; // whole warp past end

  // Build a lightweight context (assumes computeBucket exists in context)
  GpuHashMapContext<KeyT, ValueT> ctx;
  ctx.initializeDevice(num_buckets, d_keys, nullptr, d_status); // initializeDevice is expected in context

  const bool has_work = (tid < num_keys);
  const KeyT my_key = has_work ? input_keys[tid] : KeyT{0};
  const uint32_t bucket = ctx.computeBucket(my_key);

  ctx.deleteKey(has_work, laneId, my_key, bucket);
}
