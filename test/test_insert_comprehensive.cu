/*
 * Comprehensive Insert Function Test
*/

#include "gpu_hash_map.cuh"
#include <iostream>
#include <vector>
#include <numeric>  // for iota
#include <random>
#include "../src/kernels/dump_kernels.cuh"
#include <unistd.h> // Required for sleep()
#include <unordered_set>
#include <chrono>
using KeyT = uint32_t;
using ValueT = uint32_t;

int main() {
  std::cout << "=== Comprehensive Insert Function Test ===" << std::endl << std::endl;

  // Check CUDA device
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::cerr << "Error: No CUDA device available" << std::endl;
    return 1;
  }

  const uint32_t num_buckets = 1000;
  const uint32_t device_idx = 0;
  bool all_passed = true;

  // Helper for allocation and copy
  auto copy_to_device = [](const std::vector<KeyT>& h_keys,
                           const std::vector<ValueT>& h_values,
                           KeyT*& d_keys, ValueT*& d_values) {
    CHECK_CUDA_ERROR(cudaMalloc(&d_keys, h_keys.size() * sizeof(KeyT)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values, h_values.size() * sizeof(ValueT)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(),
                                h_keys.size() * sizeof(KeyT),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values.data(),
                                h_values.size() * sizeof(ValueT),
                                cudaMemcpyHostToDevice));
  };

  // ========== Test 1: Simple insertion ==========
  std::cout << "Test 1: Simple insertion" << std::endl;
  {
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 12345, false);
    std::vector<KeyT> h_keys = {1, 2, 3};
    std::vector<ValueT> h_values = {10, 20, 30};

    KeyT* d_keys; ValueT* d_values;
    copy_to_device(h_keys, h_values, d_keys, d_values);

    map.buildTable(d_keys, d_values, h_keys.size());
    uint32_t count = map.countTable();

    if (count == h_keys.size()) {
      std::cout << "  ✓ Simple insert: " << count << " elements inserted" << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected " << h_keys.size() << ", got " << count << std::endl;
      all_passed = false;
    }

    cudaFree(d_keys); cudaFree(d_values);
  }
  std::cout << std::endl;

  // ========== Test 2: Duplicate key overwrite ==========
  std::cout << "Test 2: Duplicate key overwrite" << std::endl;
  {
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 54321, false);
    std::vector<KeyT> h_keys = {5, 5, 5};
    std::vector<ValueT> h_values = {10, 20, 30};

    KeyT* d_keys; ValueT* d_values;
    copy_to_device(h_keys, h_values, d_keys, d_values);

    map.buildTable(d_keys, d_values, h_keys.size());
    uint32_t count = map.countTable();

    if (count == 1) {
      std::cout << "  ✓ Duplicate handled correctly (count = 1)" << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected 1, got " << count << std::endl;
      all_passed = false;
    }

    cudaFree(d_keys); cudaFree(d_values);
  }
  std::cout << std::endl;

  // ========== Test 3: High collision scenario ==========
  std::cout << "Test 3: High collision scenario" << std::endl;
  {
    const uint32_t small_buckets = 8;
    // enforce collision 
    GpuHashMap<KeyT, ValueT> map(int( 0.5 * small_buckets), device_idx, 123, false);
    const uint32_t num_keys = 64;

    std::vector<KeyT> h_keys(num_keys);
    std::vector<ValueT> h_values(num_keys);
    for (uint32_t i = 0; i < num_keys; i++) {
      h_keys[i] = i * small_buckets; // Force collisions
      h_values[i] = i;
    }

    KeyT* d_keys; ValueT* d_values;
    copy_to_device(h_keys, h_values, d_keys, d_values);

    map.buildTable(d_keys, d_values, num_keys);
    uint32_t count = map.countTable();

    if (count <= small_buckets) {
      std::cout << "  ✓ Handled collisions safely (count = " << count << ")" << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Table overflow or corruption" << std::endl;
      all_passed = false;
    }

    cudaFree(d_keys); cudaFree(d_values);
  }
  std::cout << std::endl;

  
  // ========== Test 4: Large random insert ==========
  std::cout << "Test 4A: Large random insert (unique count check)" << std::endl;
  {
    const uint32_t large_buckets = 1 << 20;
    int num_keys = 500000;
    std::vector<KeyT> h_keys(num_keys);
    std::vector<ValueT> h_values(num_keys);
    std::mt19937 rng(11);
    std::uniform_int_distribution<uint32_t> dist(1, 1 << 30);

    // Generate random keys
    for (uint32_t i = 0; i < num_keys; i++) {
        h_keys[i] = dist(rng);
        h_values[i] = i;
    }

    // Compute unique key count on host
    std::unordered_set<KeyT> unique_set(h_keys.begin(), h_keys.end());
    uint32_t num_unique_keys = static_cast<uint32_t>(unique_set.size());

    // Copy to device
    KeyT* d_keys; ValueT* d_values;
    copy_to_device(h_keys, h_values, d_keys, d_values);
    cudaDeviceSynchronize();
    GpuHashMap<KeyT, ValueT> map(large_buckets, device_idx, 98765, false);
    map.buildTable(d_keys, d_values, num_keys);
    cudaDeviceSynchronize();
    cudaGetLastError();

    uint32_t count = map.countTable();

    if (count == num_unique_keys) {
        std::cout << "  ✓ Passed: inserted " << num_keys 
                  << " random keys (" << num_unique_keys << " unique)" << std::endl;
    } else {
        std::cout << "  ✗ FAILED: expected " << num_unique_keys 
                  << " unique entries, got " << count << std::endl;
    }

    cudaFree(d_keys);
    cudaFree(d_values);
    std::cout << std::endl;
  }
  
  std::cout << "Test 4B: Large sequential insert (no duplicates)" << std::endl;
  {
    const uint32_t large_buckets = 1 << 20;
    int num_keys = 500000;

    std::vector<KeyT> h_keys(num_keys);
    std::vector<ValueT> h_values(num_keys);

    // Sequential unique keys
    std::iota(h_keys.begin(), h_keys.end(), 1);
    std::iota(h_values.begin(), h_values.end(), 0);

    // Copy to device
    KeyT* d_keys; ValueT* d_values;
    copy_to_device(h_keys, h_values, d_keys, d_values);
    cudaDeviceSynchronize();
    GpuHashMap<KeyT, ValueT> map(large_buckets, device_idx, 98765, false);
    map.buildTable(d_keys, d_values, num_keys);
    cudaDeviceSynchronize();
    cudaGetLastError();
    
    uint32_t count = map.countTable();

    if (count == num_keys) {
        std::cout << "  ✓ Passed: inserted " << count << " unique sequential keys" << std::endl;
    } else {
        std::cout << "  ✗ FAILED: expected " << num_keys << ", got " << count << std::endl;
    }

    cudaFree(d_keys);
    cudaFree(d_values);
    std::cout << std::endl;
  }

  // ========== Test 5: Parallel duplicate insert (overlapping keys) ==========
  std::cout << "Test 5: Parallel duplicate insert (overlapping keys)" << std::endl;
  {
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 333, false);
    const uint32_t num_keys = 1000;
    std::vector<KeyT> h_keys(num_keys);
    std::vector<ValueT> h_values(num_keys);

    for (uint32_t i = 0; i < num_keys; i++) {
      h_keys[i] = i % 100;  // repeats every 100
      h_values[i] = i;
    }

    KeyT* d_keys; ValueT* d_values;
    copy_to_device(h_keys, h_values, d_keys, d_values);

    map.buildTable(d_keys, d_values, num_keys);
    uint32_t count = map.countTable();

    if (count == 100) {
      std::cout << "  ✓ Duplicate range handled (unique count = 100)" << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected 100 unique keys, got " << count << std::endl;
      all_passed = false;
    }

    cudaFree(d_keys); cudaFree(d_values);
  }
  std::cout << std::endl;

  // ========== Test 6: Full table overflow ==========
  std::cout << "Test 6: Full table overflow" << std::endl;
  {
    const uint32_t small_buckets = 16;
    const uint32_t num_keys = 64;
    // enforce overflowing 
    GpuHashMap<KeyT, ValueT> map(int( 0.5 *small_buckets), device_idx, 777, false);

    std::vector<KeyT> h_keys(num_keys);
    std::vector<ValueT> h_values(num_keys);
    std::iota(h_keys.begin(), h_keys.end(), 0);
    std::iota(h_values.begin(), h_values.end(), 1000);

    KeyT* d_keys; ValueT* d_values;
    copy_to_device(h_keys, h_values, d_keys, d_values);

    map.buildTable(d_keys, d_values, num_keys);
    uint32_t count = map.countTable();

    if (count <= small_buckets) {
      std::cout << "  ✓ Overflow handled safely (count = " << count << ")" << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Table overfilled or corrupted" << std::endl;
      all_passed = false;
    }

    cudaFree(d_keys); cudaFree(d_values);
  }
  std::cout << std::endl;

  // ========== Final Results ==========
  if (all_passed) {
    std::cout << "=== All Insert Tests Passed! ✓ ===" << std::endl;
    return 0;
  } else {
    std::cout << "=== Some Insert Tests Failed! ✗ ===" << std::endl;
    return 1;
  }
}
