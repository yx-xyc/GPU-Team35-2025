
#include "gpu_hash_map.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cassert>

using KeyT = uint32_t;
using ValT = uint32_t;

int main() {
  std::cout << "=== Delete Test (functional) ===\n";

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "No CUDA device in CI, skip.\n";
    return 0;
  }
  CHECK_CUDA_ERROR(cudaSetDevice(0));

  const uint32_t buckets = 1u << 20; // plenty of space
  GpuHashMap<KeyT, ValT> map(buckets, 0, 123, false);

  // Prepare keys/values
  const uint32_t N = 1 << 20;
  std::vector<KeyT> h_keys(N);
  std::vector<ValT> h_vals(N);
  for (uint32_t i = 0; i < N; ++i) { h_keys[i] = i+1; h_vals[i] = i; }

  // Insert
  map.insert(h_keys.data(), h_vals.data(), N);

  // Count before delete
  uint32_t before = map.count();
  std::cout << "count before: " << before << std::endl;

  // Delete half of them (odd keys)
  std::vector<KeyT> h_del;
  h_del.reserve(N/2);
  for (uint32_t i = 0; i < N; ++i) if ((h_keys[i] & 1u) == 1u) h_del.push_back(h_keys[i]);
  map.erase(h_del.data(), (uint32_t)h_del.size());

  // Count after delete
  uint32_t after = map.count();
  std::cout << "count after: " << after << std::endl;

  // Expect about N/2 remaining
  if (!(after <= before && after >= before - N/2 - N/100)) {
    std::cerr << "Delete did not reduce count as expected.\n";
    return 1;
  }

  // Query deleted keys should be not found
  std::vector<KeyT> q = h_del;
  std::vector<ValT> res(q.size(), 0);
  uint32_t not_found = map.search(q.data(), res.data(), (uint32_t)q.size());
  if (!(not_found >= q.size() * 0.9)) {
    std::cerr << "Too many deleted keys still reported as found.\n";
    return 1;
  }

  std::cout << "✓ delete test passed\n";
  return 0;
}
