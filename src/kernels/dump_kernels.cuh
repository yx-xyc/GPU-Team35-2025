#pragma once
#include "../hash_map_context.cuh"

template <typename KeyT, typename ValueT>
__global__ void dump_table_kernel(
    GpuHashMapContext<KeyT, ValueT> ctx,
    KeyT* out_keys,
    ValueT* out_vals,
    uint32_t* out_status)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ctx.getNumBuckets()) {
        out_keys[i] = ctx.getKeys()[i];
        out_vals[i] = ctx.getValues()[i];
        out_status[i] = ctx.getStatus()[i];
    }
}
