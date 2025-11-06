/*
 * Minimal Count Test - Without Insert/Delete
 * 只测试 count 功能，不依赖 insert/delete
 */

#include "gpu_hash_map.cuh"
#include <iostream>

using KeyT = uint32_t;
using ValueT = uint32_t;

int main() {
  std::cout << "=== Minimal Count Function Test ===" << std::endl << std::endl;

  // Check CUDA device
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    std::cerr << "CUDA device check failed: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }
  if (device_count == 0) {
    std::cerr << "Error: No CUDA-capable device detected" << std::endl;
    return 1;
  }

  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    std::cerr << "Cannot set CUDA device: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  const uint32_t device_idx = 0;

  // ============================================================================
  // Test 1: Empty table with small size
  // ============================================================================
  std::cout << "Test 1: Empty table (small size)" << std::endl;
  {
    const uint32_t num_buckets = 100;
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 12345, false);
    
    uint32_t count = map.countTable();
    
    if (count == 0) {
      std::cout << "  ✓ Count = " << count << " (expected 0)" << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected 0, got " << count << std::endl;
      return 1;
    }
  }
  std::cout << std::endl;

  // ============================================================================
  // Test 2: Empty table with medium size
  // ============================================================================
  std::cout << "Test 2: Empty table (medium size)" << std::endl;
  {
    const uint32_t num_buckets = 10000;
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 12345, false);
    
    uint32_t count = map.countTable();
    
    if (count == 0) {
      std::cout << "  ✓ Count = " << count << " (expected 0)" << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected 0, got " << count << std::endl;
      return 1;
    }
  }
  std::cout << std::endl;

  // ============================================================================
  // Test 3: Empty table with large size
  // ============================================================================
  std::cout << "Test 3: Empty table (large size)" << std::endl;
  {
    const uint32_t num_buckets = 1000000;
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 12345, false);
    
    uint32_t count = map.countTable();
    
    if (count == 0) {
      std::cout << "  ✓ Count = " << count << " (expected 0)" << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected 0, got " << count << std::endl;
      return 1;
    }
  }
  std::cout << std::endl;

  // ============================================================================
  // Test 4: Count called multiple times (idempotence)
  // ============================================================================
  std::cout << "Test 4: Multiple count calls on same table" << std::endl;
  {
    const uint32_t num_buckets = 1000;
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 12345, false);
    
    uint32_t count1 = map.countTable();
    uint32_t count2 = map.countTable();
    uint32_t count3 = map.countTable();
    
    if (count1 == 0 && count2 == 0 && count3 == 0) {
      std::cout << "  ✓ All counts = 0 (idempotent)" << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected all 0, got " 
                << count1 << ", " << count2 << ", " << count3 << std::endl;
      return 1;
    }
  }
  std::cout << std::endl;

  // ============================================================================
  // Test 5: Count after clear (tests clear() function)
  // ============================================================================
  std::cout << "Test 5: Count after clear" << std::endl;
  {
    const uint32_t num_buckets = 1000;
    GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, 12345, false);
    
    // Clear an already empty table
    map.clear();
    
    uint32_t count = map.countTable();
    
    if (count == 0) {
      std::cout << "  ✓ Count after clear = " << count << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected 0, got " << count << std::endl;
      return 1;
    }
  }
  std::cout << std::endl;

  // ============================================================================
  // Test 6: Different hash seeds produce same count (for empty table)
  // ============================================================================
  std::cout << "Test 6: Different hash seeds" << std::endl;
  {
    const uint32_t num_buckets = 1000;
    
    GpuHashMap<KeyT, ValueT> map1(num_buckets, device_idx, 111, false);
    GpuHashMap<KeyT, ValueT> map2(num_buckets, device_idx, 222, false);
    GpuHashMap<KeyT, ValueT> map3(num_buckets, device_idx, 333, false);
    
    uint32_t count1 = map1.countTable();
    uint32_t count2 = map2.countTable();
    uint32_t count3 = map3.countTable();
    
    if (count1 == 0 && count2 == 0 && count3 == 0) {
      std::cout << "  ✓ All empty tables count = 0 regardless of seed" << std::endl;
    } else {
      std::cout << "  ✗ FAILED: Expected all 0, got " 
                << count1 << ", " << count2 << ", " << count3 << std::endl;
      return 1;
    }
  }
  std::cout << std::endl;

  std::cout << "=== All Count Tests Passed! ✓ ===" << std::endl;
  std::cout << std::endl;
  std::cout << "Note: Full testing requires insert/delete functionality" << std::endl;
  
  return 0;
}