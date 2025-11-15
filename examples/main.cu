/*
 * GPU Hash Map Library - Example Usage
 *
 * This example demonstrates the core features of the GPU hash map library:
 *   1. Creating a hash map with custom configuration
 *   2. Bulk insert (buildTable)
 *   3. Bulk search (searchTable) with hybrid strategy
 *   4. Bulk delete (deleteTable)
 *   5. Counting elements (countTable)
 *   6. Iterator for sequential traversal
 *
 * Build and run:
 *   cd build && cmake .. && make example && ./bin/example
 */

#include "gpu_hash_map.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using KeyT = uint32_t;
using ValueT = uint32_t;

/*
 * Helper: Simple timer for measuring performance
 */
class Timer {
 private:
  std::chrono::high_resolution_clock::time_point start_;

 public:
  void start() { start_ = std::chrono::high_resolution_clock::now(); }

  double elapsed_ms() {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
  }
};

/*
 * Helper: Print section header
 */
void printSection(const std::string& title) {
  std::cout << "\n=== " << title << " ===" << std::endl;
}

/*
 * Helper: Print performance result
 */
void printPerformance(const std::string& operation, uint32_t count, double time_ms) {
  std::cout << "  " << operation << ": " << std::fixed << std::setprecision(2)
            << time_ms << " ms"
            << " (" << std::setprecision(3) << (count / time_ms / 1000.0) << " M ops/sec)"
            << std::endl;
}

int main() {
  std::cout << "\n╔════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   GPU Hash Map Library - Example Usage    ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════╝" << std::endl;

  // ============================================================================
  // Configuration
  // ============================================================================
  const uint32_t num_keys = 1000000;      // 1M keys
  const uint32_t num_buckets = 2000000;   // 2M buckets (load factor = 0.5)
  const uint32_t device_idx = 0;          // GPU device
  const uint32_t seed = 12345;            // Random seed for hash function

  printSection("Configuration");
  std::cout << "  Number of keys:    " << num_keys << std::endl;
  std::cout << "  Number of buckets: " << num_buckets << std::endl;
  std::cout << "  Load factor:       " << std::fixed << std::setprecision(2)
            << (float)num_keys / num_buckets << std::endl;
  std::cout << "  GPU device:        " << device_idx << std::endl;

  // ============================================================================
  // Step 1: Create Hash Map
  // ============================================================================
  printSection("Step 1: Create Hash Map");
  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, device_idx, seed, false);

  std::cout << "  Hash map created successfully" << std::endl;
  std::cout << "  Memory allocated: "
            << (num_buckets * (sizeof(KeyT) + sizeof(ValueT) + sizeof(uint32_t))) / (1024.0 * 1024.0)
            << " MB" << std::endl;

  // ============================================================================
  // Step 2: Prepare Test Data
  // ============================================================================
  printSection("Step 2: Prepare Test Data");

  // Generate sequential keys (1, 2, 3, ..., num_keys) to avoid duplicates
  std::vector<KeyT> h_keys(num_keys);
  std::vector<ValueT> h_values(num_keys);

  for (uint32_t i = 0; i < num_keys; i++) {
    h_keys[i] = i + 1;          // Keys: 1, 2, 3, ..., 1000000 (no duplicates)
    h_values[i] = i * 100;      // Values: 0, 100, 200, ... (easy to verify)
  }

  std::cout << "  Generated " << num_keys << " unique sequential key-value pairs" << std::endl;
  std::cout << "  Key range: [1, " << num_keys << "]" << std::endl;

  // Allocate and copy to device
  KeyT* d_keys;
  ValueT* d_values;
  ValueT* d_results;

  CHECK_CUDA_ERROR(cudaMalloc(&d_keys, num_keys * sizeof(KeyT)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_values, num_keys * sizeof(ValueT)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_results, num_keys * sizeof(ValueT)));

  CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(KeyT),
                               cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values.data(), num_keys * sizeof(ValueT),
                               cudaMemcpyHostToDevice));

  std::cout << "  Data copied to GPU" << std::endl;

  // ============================================================================
  // Step 3: Bulk Insert (buildTable)
  // ============================================================================
  printSection("Step 3: Bulk Insert");

  Timer timer;
  timer.start();
  hash_map.buildTable(d_keys, d_values, num_keys);
  double build_time = timer.elapsed_ms();

  printPerformance("Insert", num_keys, build_time);

  // ============================================================================
  // Step 4: Count Elements
  // ============================================================================
  printSection("Step 4: Count Elements");

  uint32_t count = hash_map.countTable();
  std::cout << "  Elements in table: " << count << " / " << num_keys
            << " (" << std::fixed << std::setprecision(1)
            << (100.0 * count / num_keys) << "%)" << std::endl;

  // ============================================================================
  // Step 5: Bulk Search (searchTable with hybrid strategy)
  // ============================================================================
  printSection("Step 5: Bulk Search");

  timer.start();
  hash_map.searchTable(d_keys, d_results, num_keys);
  double search_time = timer.elapsed_ms();

  printPerformance("Search", num_keys, search_time);

  // Verify search correctness
  std::vector<ValueT> h_results(num_keys);
  CHECK_CUDA_ERROR(cudaMemcpy(h_results.data(), d_results, num_keys * sizeof(ValueT),
                               cudaMemcpyDeviceToHost));

  uint32_t num_found = 0;
  for (uint32_t i = 0; i < num_keys; i++) {
    if (h_results[i] == h_values[i]) num_found++;
  }
  std::cout << "  Verification: " << num_found << " / " << num_keys << " correct"
            << (num_found == num_keys ? " ✓" : " ✗") << std::endl;

  // ============================================================================
  // Step 6: Bulk Delete (deleteTable)
  // ============================================================================
  printSection("Step 6: Bulk Delete");

  const uint32_t num_deletes = num_keys / 2;  // Delete half the keys

  timer.start();
  hash_map.deleteTable(d_keys, num_deletes);
  double delete_time = timer.elapsed_ms();

  printPerformance("Delete", num_deletes, delete_time);

  // Count after deletion
  count = hash_map.countTable();
  std::cout << "  Remaining: " << count << " / " << num_keys
            << " (expected ~" << (num_keys - num_deletes) << ")" << std::endl;

  // Verify deleted keys are not found
  hash_map.searchTable(d_keys, d_results, num_deletes);
  CHECK_CUDA_ERROR(cudaMemcpy(h_results.data(), d_results, num_deletes * sizeof(ValueT),
                               cudaMemcpyDeviceToHost));

  uint32_t num_not_found = 0;
  for (uint32_t i = 0; i < num_deletes; i++) {
    if (h_results[i] == SEARCH_NOT_FOUND) num_not_found++;
  }
  std::cout << "  Deleted keys not found: " << num_not_found << " / " << num_deletes
            << (num_not_found == num_deletes ? " ✓" : " ✗") << std::endl;

  // ============================================================================
  // Step 7: Iterator (Sequential Traversal)
  // ============================================================================
  printSection("Step 7: Iterator for Sequential Traversal");

  timer.start();
  auto iter = hash_map.getIterator();
  double iter_time = timer.elapsed_ms();

  std::cout << "  Iterator created in: " << std::fixed << std::setprecision(2)
            << iter_time << " ms" << std::endl;
  std::cout << "  Total entries: " << iter.size() << std::endl;

  // Show first 5 entries
  std::cout << "\n  First 5 entries:" << std::endl;
  for (uint32_t i = 0; i < 5 && iter.hasNext(); i++) {
    auto pair = iter.next();
    std::cout << "    [" << i << "] Key: " << std::setw(10) << pair.key
              << " -> Value: " << pair.value << std::endl;
  }

  // Verify iterator count matches
  iter.reset();
  uint32_t iter_count = 0;
  while (iter.hasNext()) {
    iter.next();
    iter_count++;
  }
  std::cout << "\n  Iterator count matches: " << (iter_count == iter.size() ? "✓" : "✗") << std::endl;

  // ============================================================================
  // Cleanup
  // ============================================================================
  printSection("Cleanup");

  CHECK_CUDA_ERROR(cudaFree(d_keys));
  CHECK_CUDA_ERROR(cudaFree(d_values));
  CHECK_CUDA_ERROR(cudaFree(d_results));

  std::cout << "  GPU memory freed" << std::endl;

  // ============================================================================
  // Summary
  // ============================================================================
  printSection("Summary");
  std::cout << "  ✓ Hash map creation" << std::endl;
  std::cout << "  ✓ Bulk insert (" << std::setprecision(2) << (num_keys / build_time / 1000.0) << " M/s)" << std::endl;
  std::cout << "  ✓ Bulk search (" << std::setprecision(2) << (num_keys / search_time / 1000.0) << " M/s)" << std::endl;
  std::cout << "  ✓ Bulk delete (" << std::setprecision(2) << (num_deletes / delete_time / 1000.0) << " M/s)" << std::endl;
  std::cout << "  ✓ Element counting" << std::endl;
  std::cout << "  ✓ Iterator traversal" << std::endl;

  std::cout << "\n╔════════════════════════════════════════════╗" << std::endl;
  std::cout << "║       Example completed successfully!      ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════╝\n" << std::endl;

  return 0;
}
