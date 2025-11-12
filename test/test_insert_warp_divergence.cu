#include <iostream>
#include <cuda_runtime.h>
#include "gpu_hash_map.cuh"
#include "../src/kernels/dump_kernels.cuh"
#include "gpu_hash_map.cuh"
#include <iostream>
#include <vector>
#include <numeric>  // for iota
#include <random>
#include <unistd.h> // Required for sleep()
#include <unordered_set>
#include <chrono>
using KeyT = uint32_t;
using ValueT = uint32_t;

void test_divergence_pattern(GpuHashMap<KeyT, ValueT>& map, uint32_t num_keys, const std::string& pattern) {
    std::vector<KeyT> h_keys(num_keys);
    std::vector<ValueT> h_values(num_keys);
    
    if (pattern == "uniform") {
        std::iota(h_keys.begin(), h_keys.end(), 1);
    } else if (pattern == "clustered") {
        for (uint32_t i = 0; i < num_keys; ++i)
            h_keys[i] = (i % 32) + 1; // Force collisions in same warp range
    } else if (pattern == "power2") {
        for (uint32_t i = 0; i < num_keys; ++i)
            h_keys[i] = 1 << (i % 16);
    } else if (pattern == "random") {
        std::mt19937 rng(123);
        std::uniform_int_distribution<uint32_t> dist(1, 1 << 30);
        for (uint32_t i = 0; i < num_keys; ++i)
            h_keys[i] = dist(rng);
    }

    std::iota(h_values.begin(), h_values.end(), 0);

    KeyT* d_keys; ValueT* d_values;
    cudaMalloc(&d_keys, num_keys * sizeof(KeyT));
    cudaMalloc(&d_values, num_keys * sizeof(ValueT));

    cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(KeyT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values.data(), num_keys * sizeof(ValueT), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    map.buildTable(d_keys, d_values, num_keys);
    cudaDeviceSynchronize();
    cudaFree(d_keys);
    cudaFree(d_values);

    std::cout << "Ran pattern: " << pattern << std::endl;
}


int main() {
    const uint32_t num_buckets = 10000;
    const uint32_t device_idx = 0;
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 333, false);
    const uint32_t num_keys = 1000;
    test_divergence_pattern(map, num_keys, "uniform");
    // test_collision();
    // test_duplicate_update();
    // test_near_full_table();
    return 0;
}
