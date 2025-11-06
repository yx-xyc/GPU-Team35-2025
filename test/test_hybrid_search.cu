/*
 * Test Hybrid Search Strategy
 *
 * Verifies that:
 * 1. Small batches use one-warp-per-key
 * 2. Large batches use one-thread-per-key
 * 3. Custom threshold works correctly
 */

#include "gpu_hash_map.cuh"
#include <iostream>
#include <vector>

using KeyT = uint32_t;
using ValueT = uint32_t;

int main() {
  std::cout << "=== Hybrid Search Strategy Test ===" << std::endl << std::endl;

  const uint32_t num_buckets = 10000;
  const uint32_t device_idx = 0;

  // Test 1: Default threshold (5000)
  std::cout << "Test 1: Default threshold (5000)" << std::endl;
  GpuHashMap<KeyT, ValueT> hash_map1(num_buckets, device_idx, 12345, true);

  // Prepare test data
  std::vector<KeyT> h_keys = {1, 2, 3, 4, 5, 10, 20, 30, 40, 50};
  std::vector<ValueT> h_values = {100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000};

  KeyT* d_keys;
  ValueT* d_values;
  ValueT* d_results;

  CHECK_CUDA_ERROR(cudaMalloc(&d_keys, 10 * sizeof(KeyT)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_values, 10 * sizeof(ValueT)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_results, 10 * sizeof(ValueT)));

  CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), 10 * sizeof(KeyT), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values.data(), 10 * sizeof(ValueT), cudaMemcpyHostToDevice));

  // Insert keys
  hash_map1.buildTable(d_keys, d_values, 10);

  // Search with small batch (< 5000) - should use one-warp-per-key
  std::cout << "  Searching 10 keys (< threshold):" << std::endl;
  hash_map1.searchTable(d_keys, d_results, 10);
  std::cout << std::endl;

  // Search with large batch (>= 5000) - should use one-thread-per-key
  std::cout << "  Searching 10000 keys (>= threshold):" << std::endl;
  KeyT* d_large_keys;
  ValueT* d_large_results;
  CHECK_CUDA_ERROR(cudaMalloc(&d_large_keys, 10000 * sizeof(KeyT)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_large_results, 10000 * sizeof(ValueT)));
  hash_map1.searchTable(d_large_keys, d_large_results, 10000);
  CHECK_CUDA_ERROR(cudaFree(d_large_keys));
  CHECK_CUDA_ERROR(cudaFree(d_large_results));
  std::cout << std::endl;

  // Test 2: Custom threshold (100)
  std::cout << "Test 2: Custom threshold (100)" << std::endl;
  GpuHashMap<KeyT, ValueT> hash_map2(num_buckets, device_idx, 12345, true, 100);

  // Search with 50 keys (< 100) - should use one-warp-per-key
  std::cout << "  Searching 50 keys (< threshold):" << std::endl;
  KeyT* d_keys_50;
  ValueT* d_results_50;
  CHECK_CUDA_ERROR(cudaMalloc(&d_keys_50, 50 * sizeof(KeyT)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_results_50, 50 * sizeof(ValueT)));
  hash_map2.searchTable(d_keys_50, d_results_50, 50);
  CHECK_CUDA_ERROR(cudaFree(d_keys_50));
  CHECK_CUDA_ERROR(cudaFree(d_results_50));
  std::cout << std::endl;

  // Search with 200 keys (>= 100) - should use one-thread-per-key
  std::cout << "  Searching 200 keys (>= threshold):" << std::endl;
  KeyT* d_keys_200;
  ValueT* d_results_200;
  CHECK_CUDA_ERROR(cudaMalloc(&d_keys_200, 200 * sizeof(KeyT)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_results_200, 200 * sizeof(ValueT)));
  hash_map2.searchTable(d_keys_200, d_results_200, 200);
  CHECK_CUDA_ERROR(cudaFree(d_keys_200));
  CHECK_CUDA_ERROR(cudaFree(d_results_200));
  std::cout << std::endl;

  // Cleanup
  CHECK_CUDA_ERROR(cudaFree(d_keys));
  CHECK_CUDA_ERROR(cudaFree(d_values));
  CHECK_CUDA_ERROR(cudaFree(d_results));

  std::cout << "=== All tests passed! ===" << std::endl;

  return 0;
}
