/*
 * Build Kernel - Bulk Insert Operation
 *
 * Launches many threads to insert key-value pairs in parallel.
 * Each warp cooperates to insert ~32 keys.
 *
 * KERNEL PATTERN:
 *   - Each thread processes one key
 *   - Threads are grouped into warps (32 threads)
 *   - Each warp cooperates using insertKey()
 *
 * GRID/BLOCK CONFIGURATION:
 *   - Block size: 128 or 256 threads (4-8 warps per block)
 *   - Grid size: enough blocks to cover all keys
 *   - Example: For 1M keys, use 256 threads/block = ~4000 blocks
 *
 * REFERENCES:
 *   - SlabHash/src/concurrent_map/device/build.cuh
 */

#pragma once

#include "../hash_map_context.cuh"

/*
 * Kernel: Insert many key-value pairs in parallel
 *
 * Parameters:
 *   ctx - hash map context (shallow copy from host)
 *   d_keys - device array of keys to insert
 *   d_values - device array of values to insert
 *   num_keys - number of keys
 */


template <typename KeyT, typename ValueT>
__global__ void build_table_kernel(
    GpuHashMapContext<KeyT, ValueT> ctx,
    const KeyT* d_keys,
    const ValueT* d_values,
    uint32_t num_keys) 
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t laneId = threadIdx.x & 0x1F;

    // Early exit for warps beyond data (warp-level exit)
    if ((tid - laneId) >= num_keys) return;

    // Each thread checks if it has real work
    bool has_work = (tid < num_keys);
    KeyT key = has_work ? d_keys[tid] : KeyT();
    ValueT value = has_work ? d_values[tid] : ValueT();

    // Compute bucket (safe even if has_work=false)
    uint32_t bucket = ctx.computeBucket(key);


    // Warp cooperates - threads with has_work=false participate but don't insert
    ctx.insertKey(has_work, laneId, key, value, bucket);
}

/*
 * Host wrapper: Launch build kernel
 */

template <typename KeyT, typename ValueT>
void GpuHashMap<KeyT, ValueT>::buildTable(
    const KeyT* d_keys,
    const ValueT* d_values,
    uint32_t num_keys)
{
    const uint32_t block_size = 128;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;

    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    build_table_kernel<<<num_blocks, block_size>>>(context_, d_keys, d_values, num_keys);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
