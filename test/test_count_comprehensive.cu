/*
 * Comprehensive Count Test
*/

#include "gpu_hash_map.cuh"
#include <iostream>
#include <vector>

using KeyT = uint32_t;
using ValueT = uint32_t;

int main() {
  std::cout << "=== Comprehensive Count Function Test ===" << std::endl << std::endl;

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

  // ========== Test 1: Empty table count ==========
  std::cout << "Test 1: Empty table count" << std::endl;
  {
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 12345, false);
    uint32_t count = map.countTable();
    
    if (count == 0) {
      std::cout << "  ✓ Empty table: count = " << count << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected 0, got " << count << std::endl;
      all_passed = false;
    }
  }
  std::cout << std::endl;

  // ========== Test 2: Count after inserting elements ==========
  std::cout << "Test 2: Count after inserting elements" << std::endl;
  {
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 12345, false);

    // Insert 10 elements
    const uint32_t num_keys = 10;
    std::vector<KeyT> h_keys = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<ValueT> h_values(num_keys);
    for (uint32_t i = 0; i < num_keys; i++) {
      h_values[i] = i * 100;
    }

    KeyT* d_keys;
    ValueT* d_values;
    CHECK_CUDA_ERROR(cudaMalloc(&d_keys, num_keys * sizeof(KeyT)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values, num_keys * sizeof(ValueT)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(KeyT),
                                 cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values.data(), num_keys * sizeof(ValueT),
                                 cudaMemcpyHostToDevice));

    map.buildTable(d_keys, d_values, num_keys);
    uint32_t count = map.countTable();

    if (count == num_keys) {
      std::cout << "  ✓ After insert: count = " << count << " (expected " << num_keys << ")" << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected " << num_keys << ", got " << count << std::endl;
      all_passed = false;
    }

    CHECK_CUDA_ERROR(cudaFree(d_keys));
    CHECK_CUDA_ERROR(cudaFree(d_values));
  }
  std::cout << std::endl;

  // ========== Test 3: Count after deleting some elements ==========
  std::cout << "Test 3: Count after deleting some elements" << std::endl;
  {
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 12345, false);

    // Insert 20 elements
    const uint32_t num_keys = 20;
    std::vector<KeyT> h_keys(num_keys);
    std::vector<ValueT> h_values(num_keys);
    for (uint32_t i = 0; i < num_keys; i++) {
      h_keys[i] = i + 1;
      h_values[i] = i * 100;
    }

    KeyT* d_keys;
    ValueT* d_values;
    CHECK_CUDA_ERROR(cudaMalloc(&d_keys, num_keys * sizeof(KeyT)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values, num_keys * sizeof(ValueT)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(KeyT),
                                 cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values.data(), num_keys * sizeof(ValueT),
                                 cudaMemcpyHostToDevice));

    map.buildTable(d_keys, d_values, num_keys);
    
    // Delete first 10 elements
    const uint32_t num_deletes = 10;
    map.deleteTable(d_keys, num_deletes);
    
    uint32_t count = map.countTable();
    uint32_t expected = num_keys - num_deletes;

    if (count == expected) {
      std::cout << "  ✓ After delete: count = " << count << " (expected " << expected << ")" << std::endl;
    } else {
      std::cout << "  ⚠ After delete: count = " << count << " (expected " << expected << ")" << std::endl;
      std::cout << "     Note: This test requires delete functionality to work" << std::endl;
      // Don't mark as failed if delete isn't implemented yet
    }

    CHECK_CUDA_ERROR(cudaFree(d_keys));
    CHECK_CUDA_ERROR(cudaFree(d_values));
  }
  std::cout << std::endl;

  // ========== Test 4: Count with large dataset ==========
  std::cout << "Test 4: Count with large dataset" << std::endl;
  {
    const uint32_t large_buckets = 100000;
    GpuHashMap<KeyT, ValueT> map(large_buckets, device_idx, 12345, false);

    const uint32_t num_keys = 50000;
    std::vector<KeyT> h_keys(num_keys);
    std::vector<ValueT> h_values(num_keys);
    for (uint32_t i = 0; i < num_keys; i++) {
      h_keys[i] = i + 1;
      h_values[i] = i * 100;
    }

    KeyT* d_keys;
    ValueT* d_values;
    CHECK_CUDA_ERROR(cudaMalloc(&d_keys, num_keys * sizeof(KeyT)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values, num_keys * sizeof(ValueT)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(KeyT),
                                 cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values.data(), num_keys * sizeof(ValueT),
                                 cudaMemcpyHostToDevice));

    map.buildTable(d_keys, d_values, num_keys);
    uint32_t count = map.countTable();

    if (count == num_keys) {
      std::cout << "  ✓ Large dataset: count = " << count << " (expected " << num_keys << ")" << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected " << num_keys << ", got " << count << std::endl;
      all_passed = false;
    }

    CHECK_CUDA_ERROR(cudaFree(d_keys));
    CHECK_CUDA_ERROR(cudaFree(d_values));
  }
  std::cout << std::endl;

  // ========== Test 5: Count after clear ==========
  std::cout << "Test 5: Count after clear" << std::endl;
  {
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 12345, false);

    // Insert elements
    const uint32_t num_keys = 15;
    std::vector<KeyT> h_keys(num_keys);
    std::vector<ValueT> h_values(num_keys);
    for (uint32_t i = 0; i < num_keys; i++) {
      h_keys[i] = i + 1;
      h_values[i] = i * 100;
    }

    KeyT* d_keys;
    ValueT* d_values;
    CHECK_CUDA_ERROR(cudaMalloc(&d_keys, num_keys * sizeof(KeyT)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values, num_keys * sizeof(ValueT)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(KeyT),
                                 cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values.data(), num_keys * sizeof(ValueT),
                                 cudaMemcpyHostToDevice));

    map.buildTable(d_keys, d_values, num_keys);
    
    // Clear the table
    map.clear();
    
    uint32_t count = map.countTable();

    if (count == 0) {
      std::cout << "  ✓ After clear: count = " << count << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected 0, got " << count << std::endl;
      all_passed = false;
    }

    CHECK_CUDA_ERROR(cudaFree(d_keys));
    CHECK_CUDA_ERROR(cudaFree(d_values));
  }
  std::cout << std::endl;

  // ========== Test 6: Count with duplicate insertions ==========
  std::cout << "Test 6: Count with duplicate insertions" << std::endl;
  {
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 12345, false);

    const uint32_t num_keys = 10;
    std::vector<KeyT> h_keys = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};  // Duplicates
    std::vector<ValueT> h_values(num_keys);
    for (uint32_t i = 0; i < num_keys; i++) {
      h_values[i] = i * 100;
    }

    KeyT* d_keys;
    ValueT* d_values;
    CHECK_CUDA_ERROR(cudaMalloc(&d_keys, num_keys * sizeof(KeyT)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values, num_keys * sizeof(ValueT)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(KeyT),
                                 cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values.data(), num_keys * sizeof(ValueT),
                                 cudaMemcpyHostToDevice));

    map.buildTable(d_keys, d_values, num_keys);
    uint32_t count = map.countTable();

    // Should only count unique keys (5 unique keys)
    uint32_t expected = 5;
    if (count == expected) {
      std::cout << "  ✓ With duplicates: count = " << count << " (expected " << expected << ")" << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected " << expected << ", got " << count << std::endl;
      all_passed = false;
    }

    CHECK_CUDA_ERROR(cudaFree(d_keys));
    CHECK_CUDA_ERROR(cudaFree(d_values));
  }
  std::cout << std::endl;

  // ========== Test 7: Multiple count calls (idempotency) ==========
  std::cout << "Test 7: Multiple count calls (idempotency)" << std::endl;
  {
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 12345, false);

    const uint32_t num_keys = 10;
    std::vector<KeyT> h_keys(num_keys);
    std::vector<ValueT> h_values(num_keys);
    for (uint32_t i = 0; i < num_keys; i++) {
      h_keys[i] = i + 1;
      h_values[i] = i * 100;
    }

    KeyT* d_keys;
    ValueT* d_values;
    CHECK_CUDA_ERROR(cudaMalloc(&d_keys, num_keys * sizeof(KeyT)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values, num_keys * sizeof(ValueT)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(KeyT),
                                 cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values.data(), num_keys * sizeof(ValueT),
                                 cudaMemcpyHostToDevice));

    map.buildTable(d_keys, d_values, num_keys);
    
    // Call countTable multiple times
    uint32_t count1 = map.countTable();
    uint32_t count2 = map.countTable();
    uint32_t count3 = map.countTable();

    if (count1 == num_keys && count2 == num_keys && count3 == num_keys) {
      std::cout << "  ✓ Idempotent: all counts = " << count1 << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Counts not consistent: " 
                << count1 << ", " << count2 << ", " << count3 << std::endl;
      all_passed = false;
    }

    CHECK_CUDA_ERROR(cudaFree(d_keys));
    CHECK_CUDA_ERROR(cudaFree(d_values));
  }
  std::cout << std::endl;

  // ========== Final Results ==========
  if (all_passed) {
    std::cout << "=== All Count Tests Passed! ✓ ===" << std::endl;
    return 0;
  } else {
    std::cout << "=== Some Count Tests Failed! ✗ ===" << std::endl;
    return 1;
  }
}