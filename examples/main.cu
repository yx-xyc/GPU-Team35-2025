/*
 * GPU Hash Map - Example Program
 *
 * Demonstrates basic usage of the hash map library.
 * Shows how to:
 *   - Create a hash map
 *   - Insert key-value pairs
 *   - Search for keys
 *   - Delete keys
 *   - Count elements
 *   - Use iterator
 *
 * COMPILE AND RUN:
 *   mkdir build && cd build
 *   cmake ..
 *   make
 *   ./bin/example
 */

#include "gpu_hash_map.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using KeyT = uint32_t;
using ValueT = uint32_t;

/*
 * Helper: Generate random keys
 */
void generateRandomKeys(std::vector<KeyT>& keys, uint32_t num_keys, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<KeyT> dist(1, UINT32_MAX);

  keys.resize(num_keys);
  for (uint32_t i = 0; i < num_keys; i++) {
    keys[i] = dist(rng);
  }
}

/*
 * Helper: Timing utility
 */
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

int main() {
  std::cout << "=== GPU Hash Map Example ===" << std::endl << std::endl;

  // Configuration
  const uint32_t num_keys = 1000000;  // 1M keys
  const uint32_t num_buckets = 2000000;  // 2M buckets (load factor ~0.5)
  const uint32_t device_idx = 0;

  std::cout << "Configuration:" << std::endl;
  std::cout << "  Num keys: " << num_keys << std::endl;
  std::cout << "  Num buckets: " << num_buckets << std::endl;
  std::cout << "  Load factor: " << (float)num_keys / num_buckets << std::endl;
  std::cout << std::endl;

  // TODO: Uncomment and implement after finishing the implementation
  /*
  // 1. Create hash map
  std::cout << "Creating hash map..." << std::endl;
  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, device_idx, 12345, true);
  std::cout << std::endl;

  // 2. Generate random keys and values
  std::cout << "Generating " << num_keys << " random keys..." << std::endl;
  std::vector<KeyT> h_keys;
  generateRandomKeys(h_keys, num_keys, 42);

  std::vector<ValueT> h_values(num_keys);
  for (uint32_t i = 0; i < num_keys; i++) {
    h_values[i] = i;  // Value = index
  }

  // Copy to device
  KeyT* d_keys;
  ValueT* d_values;
  CHECK_CUDA_ERROR(cudaMalloc(&d_keys, num_keys * sizeof(KeyT)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_values, num_keys * sizeof(ValueT)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(KeyT),
                               cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values.data(), num_keys * sizeof(ValueT),
                               cudaMemcpyHostToDevice));

  // 3. Insert (build table)
  std::cout << "Inserting " << num_keys << " keys..." << std::endl;
  Timer timer;
  timer.start();
  hash_map.buildTable(d_keys, d_values, num_keys);
  double build_time = timer.elapsed_ms();
  std::cout << "  Time: " << build_time << " ms" << std::endl;
  std::cout << "  Throughput: " << (num_keys / build_time / 1000.0) << " M keys/sec" << std::endl;
  std::cout << std::endl;

  // 4. Count elements
  std::cout << "Counting elements..." << std::endl;
  uint32_t count = hash_map.countTable();
  std::cout << "  Count: " << count << " / " << num_keys << std::endl;
  std::cout << std::endl;

  // 5. Search for keys
  std::cout << "Searching for " << num_keys << " keys..." << std::endl;
  ValueT* d_results;
  CHECK_CUDA_ERROR(cudaMalloc(&d_results, num_keys * sizeof(ValueT)));

  timer.start();
  hash_map.searchTable(d_keys, d_results, num_keys);
  double search_time = timer.elapsed_ms();
  std::cout << "  Time: " << search_time << " ms" << std::endl;
  std::cout << "  Throughput: " << (num_keys / search_time / 1000.0) << " M keys/sec" << std::endl;

  // Verify results
  std::vector<ValueT> h_results(num_keys);
  CHECK_CUDA_ERROR(cudaMemcpy(h_results.data(), d_results, num_keys * sizeof(ValueT),
                               cudaMemcpyDeviceToHost));

  uint32_t num_found = 0;
  for (uint32_t i = 0; i < num_keys; i++) {
    if (h_results[i] == i) {
      num_found++;
    }
  }
  std::cout << "  Found: " << num_found << " / " << num_keys << std::endl;
  std::cout << std::endl;

  // 6. Delete half the keys
  uint32_t num_deletes = num_keys / 2;
  std::cout << "Deleting " << num_deletes << " keys..." << std::endl;
  timer.start();
  hash_map.deleteTable(d_keys, num_deletes);
  double delete_time = timer.elapsed_ms();
  std::cout << "  Time: " << delete_time << " ms" << std::endl;
  std::cout << "  Throughput: " << (num_deletes / delete_time / 1000.0) << " M keys/sec" << std::endl;
  std::cout << std::endl;

  // 7. Count after deletion
  std::cout << "Counting after deletion..." << std::endl;
  count = hash_map.countTable();
  std::cout << "  Count: " << count << " (expected ~" << (num_keys - num_deletes) << ")" << std::endl;
  std::cout << std::endl;

  // 8. Search for deleted keys
  std::cout << "Searching for deleted keys..." << std::endl;
  hash_map.searchTable(d_keys, d_results, num_deletes);
  CHECK_CUDA_ERROR(cudaMemcpy(h_results.data(), d_results, num_deletes * sizeof(ValueT),
                               cudaMemcpyDeviceToHost));

  uint32_t num_not_found = 0;
  for (uint32_t i = 0; i < num_deletes; i++) {
    if (h_results[i] == SEARCH_NOT_FOUND) {
      num_not_found++;
    }
  }
  std::cout << "  Not found: " << num_not_found << " / " << num_deletes << std::endl;
  std::cout << std::endl;

  // Cleanup
  CHECK_CUDA_ERROR(cudaFree(d_keys));
  CHECK_CUDA_ERROR(cudaFree(d_values));
  CHECK_CUDA_ERROR(cudaFree(d_results));
  */

  std::cout << "TODO: Complete implementation of hash map operations" << std::endl;
  std::cout << "Once implemented, uncomment the code above to run the example." << std::endl;

  return 0;
}
