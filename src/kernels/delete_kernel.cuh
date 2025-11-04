#pragma once
#include "../warp/delete.cuh"
#include <cuda_runtime.h>

// GPU kernel: 每个线程删除一个 key
template <typename KeyT, typename ValueT>
__global__ void delete_table_kernel(
    KeyT* d_keys,
    uint32_t* d_status,
    uint32_t num_buckets,
    const KeyT* input_keys,
    uint32_t num_keys) {

  uint32_t tid    = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_keys) return;

  bool has_work = (tid < num_keys);
  KeyT key{};
  uint32_t bucket = 0;

  if (has_work) {
    key = input_keys[tid];
    bucket = ((uint64_t)key * 11400714819323198485ull) % num_buckets;
  }

  deleteKey<KeyT, ValueT>(
      d_keys, d_status, num_buckets, has_work, laneId, key, bucket);
}

template <typename KeyT, typename ValueT>
inline void launch_delete_kernel(
    KeyT* d_keys,
    uint32_t* d_status,
    uint32_t num_buckets,
    const KeyT* input_keys,
    uint32_t num_keys,
    cudaStream_t stream = nullptr) {

  if (num_keys == 0) return;

  const uint32_t block = 128;
  const uint32_t grid  = (num_keys + block - 1) / block;

  delete_table_kernel<KeyT, ValueT><<<grid, block, 0, stream>>>(
      d_keys, d_status, num_buckets, input_keys, num_keys);
}

template <typename KeyT, typename ValueT>
void GpuHashMap<KeyT, ValueT>::deleteTable(const KeyT* d_keys, uint32_t num_keys) {
  launch_delete_kernel<KeyT, ValueT>(context_.getKeys(), context_.getStatus(), context_.getNumBuckets(), d_keys, num_keys);
}