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

void perf_delete(uint32_t num_keys, uint32_t num_buckets) {
  std::cout << "\n============================================" << std::endl;
  std::cout << "Keys: " << num_keys << " Buckets: " << num_buckets
            << "  (Load factor: " << std::fixed << std::setprecision(2)
            << (float)num_keys / num_buckets << ")" << std::endl;
  std::cout << "============================================" << std::endl;

  GpuHashMap<KeyT, ValueT> map(num_buckets, 0, 12345, false);

  std::vector<KeyT> h_keys(num_keys);
  std::vector<ValueT> h_vals(num_keys);
  for (uint32_t i = 0; i < num_keys; i++) {
    h_keys[i] = i + 1;
    h_vals[i] = i;
  }

  KeyT *d_keys;
  ValueT *d_vals;
  cudaMalloc(&d_keys, num_keys * sizeof(KeyT));
  cudaMalloc(&d_vals, num_keys * sizeof(ValueT));
  cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(KeyT), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vals, h_vals.data(), num_keys * sizeof(ValueT), cudaMemcpyHostToDevice);

  map.buildTable(d_keys, d_vals, num_keys);

  // Warm-up
  map.deleteTable(d_keys, num_keys / 2);
  map.buildTable(d_keys, d_vals, num_keys);
  map.deleteTableOptimized(d_keys, num_keys / 2);
  map.buildTable(d_keys, d_vals, num_keys);

  Timer timer;
  const int rounds = 10;

  // Baseline DELETE
  double total_init = 0;
  for (int i = 0; i < rounds; i++) {
    map.buildTable(d_keys, d_vals, num_keys);
    timer.start();
    map.deleteTable(d_keys, num_keys / 2);
    cudaDeviceSynchronize();
    total_init += timer.elapsed_ms();
  }
  double avg_init = total_init / rounds;

  // Optimized DELETE
  double total_opt = 0;
  for (int i = 0; i < rounds; i++) {
    map.buildTable(d_keys, d_vals, num_keys);
    timer.start();
    map.deleteTableOptimized(d_keys, num_keys / 2);
    cudaDeviceSynchronize();
    total_opt += timer.elapsed_ms();
  }
  double avg_opt = total_opt / rounds;

  double speedup = avg_init / avg_opt;
  double improvement = (1 - avg_opt / avg_init) * 100.0;

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Initial delete:     " << avg_init << " ms" << std::endl;
  std::cout << "Optimized delete:   " << avg_opt << " ms" << std::endl;
  std::cout << "Speedup:            " << speedup << "x" << std::endl;
  std::cout << "Improvement:        " << improvement << "%" << std::endl;

  cudaFree(d_keys);
  cudaFree(d_vals);
}

int main() {
  std::cout << "===== Delete Performance Test =====" << std::endl;

  perf_delete(1000000, 2000000);
  perf_delete(5000000, 10000000);
  perf_delete(10000000, 20000000);
  perf_delete(20000000, 40000000);
  perf_delete(50000000, 100000000);

  std::cout << "===== Test Completed =====" << std::endl;
  return 0;
}

