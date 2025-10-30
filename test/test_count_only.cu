/*
 * Minimal Test - Count Only
 * 测试 count 功能是否正常工作
 */

#include "gpu_hash_map.cuh"
#include <iostream>

int main() {
  std::cout << "=== Testing Count Function ===" << std::endl;

  const uint32_t num_buckets = 100;
  
  // 创建空哈希表
  GpuHashMap<uint32_t, uint32_t> map(num_buckets, 0, 12345, true);

  // 测试空表的 count
  std::cout << "Testing empty table count..." << std::endl;
  uint32_t count = map.countTable();
  std::cout << "Count: " << count << std::endl;
  
  if (count == 0) {
    std::cout << "✓ Count test passed!" << std::endl;
    return 0;
  } else {
    std::cout << "✗ Count test failed! Expected 0, got " << count << std::endl;
    return 1;
  }
}