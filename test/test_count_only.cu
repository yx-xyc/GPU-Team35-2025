/*
 * Minimal Test - Count Only
 * 测试 count 功能是否正常工作
 */

#include "gpu_hash_map.cuh"
#include <iostream>

int main() {
  std::cout << "=== Testing Count Function ===" << std::endl;

  const uint32_t num_buckets = 100;

  int device_count = 0;

  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
      printf("CUDA 设备检查失败：%s（错误码：%d）\n", cudaGetErrorString(err), err);
      return 1;
  }
  if (device_count == 0) {
      printf("错误：未检测到可用的 CUDA 设备（no CUDA-capable device is detected）\n");
      return 1;
  }

  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
      printf("无法设置 CUDA 设备：%s（错误码：%d）\n", cudaGetErrorString(err), err);
      return 1;
  }


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