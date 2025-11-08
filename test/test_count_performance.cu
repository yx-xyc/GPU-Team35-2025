/*
 * Count Performance Comparison Test
 */

#include "gpu_hash_map.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using KeyT = uint32_t;
using ValueT = uint32_t;

class Timer {
private:
  std::chrono::high_resolution_clock::time_point start_;
public:
  void start() {
    start_ = std::chrono::high_resolution_clock::now();
  }
  double elapsed_ms() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start_;
    return elapsed.count();
  }
};

void test_count_performance(uint32_t num_keys, uint32_t num_buckets) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "Testing with " << num_keys << " keys, " 
            << num_buckets << " buckets" << std::endl;
  std::cout << "Load factor: " << std::fixed << std::setprecision(2) 
            << (float)num_keys / num_buckets << std::endl;
  std::cout << "========================================" << std::endl;

  GpuHashMap<KeyT, ValueT> map(num_buckets, 0, 12345, false);

  // Generate and insert keys
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
  CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), 
                               num_keys * sizeof(KeyT),
                               cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values.data(), 
                               num_keys * sizeof(ValueT),
                               cudaMemcpyHostToDevice));

  map.buildTable(d_keys, d_values, num_keys);

  // Warm up
  map.countTable();
  map.countTableOptimized();

  const int num_runs = 20;
  Timer timer;

  // Test initial version
  std::cout << "\n--- INITIAL version ---" << std::endl;
  double total_initial = 0;
  uint32_t result_initial = 0;
  
  for (int i = 0; i < num_runs; i++) {
    timer.start();
    result_initial = map.countTable();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    total_initial += timer.elapsed_ms();
  }
  double avg_initial = total_initial / num_runs;
  std::cout << "Count: " << result_initial << std::endl;
  std::cout << "Time:  " << std::fixed << std::setprecision(4) 
            << avg_initial << " ms" << std::endl;

  // Test optimized version
  std::cout << "\n--- OPTIMIZED version ---" << std::endl;
  double total_opt = 0;
  uint32_t result_opt = 0;
  
  for (int i = 0; i < num_runs; i++) {
    timer.start();
    result_opt = map.countTableOptimized();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    total_opt += timer.elapsed_ms();
  }
  double avg_opt = total_opt / num_runs;
  std::cout << "Count: " << result_opt << std::endl;
  std::cout << "Time:  " << std::fixed << std::setprecision(4) 
            << avg_opt << " ms" << std::endl;

  // Comparison
  std::cout << "\n--- PERFORMANCE ---" << std::endl;
  double speedup = avg_initial / avg_opt;
  double improvement = (1.0 - avg_opt / avg_initial) * 100;
  
  std::cout << "Speedup:     " << std::fixed << std::setprecision(2) 
            << speedup << "x" << std::endl;
  std::cout << "Improvement: " << std::fixed << std::setprecision(1)
            << improvement << "%" << std::endl;

  if (result_initial != result_opt || result_initial != num_keys) {
    std::cout << "\n⚠ WARNING: Results don't match!" << std::endl;
  } else {
    std::cout << "\n✓ Results verified correct" << std::endl;
  }

  CHECK_CUDA_ERROR(cudaFree(d_keys));
  CHECK_CUDA_ERROR(cudaFree(d_values));
}

int main() {
  std::cout << "======Count Performance Comparison=====" << std::endl;

  test_count_performance(1000000, 2000000);    // 1M
  test_count_performance(5000000, 10000000);   // 5M
  test_count_performance(10000000, 20000000);  // 10M
  test_count_performance(20000000, 40000000);  // 20M
  test_count_performance(50000000, 100000000);  // 50M

  std::cout << "======Tests completed!======" << std::endl;

  return 0;
}