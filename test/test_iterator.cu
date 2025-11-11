/*
 * Iterator Test Suite
 *
 * Comprehensive tests for GpuHashMapIterator functionality:
 *   1. Empty table iteration
 *   2. Basic iteration
 *   3. Iteration after deletions
 *   4. Iteration with collisions
 *   5. Large table iteration
 *   6. Iterator reset functionality
 *   7. Multiple iterations (idempotence)
 *   8. Edge cases (full table, high load factor)
 */

#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include "gpu_hash_map.cuh"
#include "test_utils.cuh"

using namespace TestUtils;

// ============================================================================
// Test 1: Empty Table Iteration
// ============================================================================
void test_empty_iteration() {
  printSectionHeader("Test 1: Empty Table Iteration");

  const uint32_t num_buckets = 1000;
  GpuHashMap<uint32_t, uint32_t> hash_map(num_buckets, 0, 42, false);

  // Get iterator for empty table
  auto iter = hash_map.getIterator();

  // Verify
  bool passed = (iter.size() == 0) && (!iter.hasNext());

  printTestResult("Empty table has 0 entries", passed);
  std::cout << "  Iterator size: " << iter.size() << std::endl;
}

// ============================================================================
// Test 2: Basic Iteration
// ============================================================================
void test_basic_iteration() {
  printSectionHeader("Test 2: Basic Iteration");

  const uint32_t num_buckets = 1000;
  const uint32_t num_keys = 10;
  GpuHashMap<uint32_t, uint32_t> hash_map(num_buckets, 0, 42, false);

  // Generate keys and values
  std::vector<uint32_t> h_keys, h_values;
  generateSequentialKeys<uint32_t>(h_keys, num_keys, 100);  // keys: 100, 101, 102, ...
  generateSequentialKeys<uint32_t>(h_values, num_keys, 1000);  // values: 1000, 1001, 1002, ...

  // Insert
  uint32_t* d_keys = allocateDeviceArray<uint32_t>(num_keys);
  uint32_t* d_values = allocateDeviceArray<uint32_t>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);
  hash_map.buildTable(d_keys, d_values, num_keys);

  // Create ground truth map
  std::unordered_map<uint32_t, uint32_t> ground_truth;
  for (uint32_t i = 0; i < num_keys; i++) {
    ground_truth[h_keys[i]] = h_values[i];
  }

  // Iterate and verify
  auto iter = hash_map.getIterator();

  bool passed = true;
  uint32_t count = 0;

  while (iter.hasNext()) {
    auto pair = iter.next();
    count++;

    // Check if key exists in ground truth
    if (ground_truth.find(pair.key) == ground_truth.end()) {
      std::cout << "  ERROR: Key " << pair.key << " not in ground truth!" << std::endl;
      passed = false;
    } else if (ground_truth[pair.key] != pair.value) {
      std::cout << "  ERROR: Value mismatch for key " << pair.key
                << " (expected " << ground_truth[pair.key]
                << ", got " << pair.value << ")" << std::endl;
      passed = false;
    }
  }

  // Verify count
  if (count != num_keys) {
    std::cout << "  ERROR: Expected " << num_keys << " entries, got " << count << std::endl;
    passed = false;
  }

  if (iter.size() != num_keys) {
    std::cout << "  ERROR: Iterator size() returned " << iter.size()
              << ", expected " << num_keys << std::endl;
    passed = false;
  }

  printTestResult("Basic iteration correctness", passed);
  std::cout << "  Entries iterated: " << count << " / " << num_keys << std::endl;

  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
}

// ============================================================================
// Test 3: Iteration After Deletions
// ============================================================================
void test_iteration_after_deletions() {
  printSectionHeader("Test 3: Iteration After Deletions");

  const uint32_t num_buckets = 1000;
  const uint32_t num_keys = 20;
  const uint32_t num_deletes = 5;
  GpuHashMap<uint32_t, uint32_t> hash_map(num_buckets, 0, 42, false);

  // Generate and insert keys
  std::vector<uint32_t> h_keys, h_values;
  generateSequentialKeys<uint32_t>(h_keys, num_keys, 200);
  generateSequentialKeys<uint32_t>(h_values, num_keys, 2000);

  uint32_t* d_keys = allocateDeviceArray<uint32_t>(num_keys);
  uint32_t* d_values = allocateDeviceArray<uint32_t>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);
  hash_map.buildTable(d_keys, d_values, num_keys);

  // Delete some keys (delete keys 200, 202, 204, 206, 208)
  std::vector<uint32_t> h_delete_keys;
  for (uint32_t i = 0; i < num_deletes; i++) {
    h_delete_keys.push_back(200 + i * 2);
  }
  uint32_t* d_delete_keys = allocateDeviceArray<uint32_t>(num_deletes);
  copyToDevice(d_delete_keys, h_delete_keys);
  hash_map.deleteTable(d_delete_keys, num_deletes);

  // Create ground truth (remaining keys)
  std::unordered_map<uint32_t, uint32_t> ground_truth;
  for (uint32_t i = 0; i < num_keys; i++) {
    // Skip deleted keys
    bool is_deleted = false;
    for (uint32_t j = 0; j < num_deletes; j++) {
      if (h_keys[i] == h_delete_keys[j]) {
        is_deleted = true;
        break;
      }
    }
    if (!is_deleted) {
      ground_truth[h_keys[i]] = h_values[i];
    }
  }

  // Iterate and verify
  auto iter = hash_map.getIterator();
  bool passed = true;
  uint32_t count = 0;

  while (iter.hasNext()) {
    auto pair = iter.next();
    count++;

    if (ground_truth.find(pair.key) == ground_truth.end()) {
      std::cout << "  ERROR: Key " << pair.key << " should have been deleted!" << std::endl;
      passed = false;
    }
  }

  uint32_t expected_count = num_keys - num_deletes;
  if (count != expected_count) {
    std::cout << "  ERROR: Expected " << expected_count << " entries, got " << count << std::endl;
    passed = false;
  }

  printTestResult("Iteration after deletions", passed);
  std::cout << "  Remaining entries: " << count << " / " << expected_count << std::endl;

  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
  freeDeviceArray(d_delete_keys);
}

// ============================================================================
// Test 4: Large Table Iteration
// ============================================================================
void test_large_table_iteration() {
  printSectionHeader("Test 4: Large Table Iteration");

  const uint32_t num_keys = 100000;
  const uint32_t num_buckets = 200000;  // Load factor 0.5
  GpuHashMap<uint32_t, uint32_t> hash_map(num_buckets, 0, 42, false);

  // Generate unique random keys
  std::vector<uint32_t> h_keys, h_values;
  generateRandomUniqueKeys<uint32_t>(h_keys, num_keys, 123);
  generateSequentialKeys<uint32_t>(h_values, num_keys, 10000);

  uint32_t* d_keys = allocateDeviceArray<uint32_t>(num_keys);
  uint32_t* d_values = allocateDeviceArray<uint32_t>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);

  Timer timer;
  timer.start();
  hash_map.buildTable(d_keys, d_values, num_keys);
  cudaDeviceSynchronize();
  double insert_time = timer.elapsedMs();

  // Create ground truth
  std::unordered_map<uint32_t, uint32_t> ground_truth;
  for (uint32_t i = 0; i < num_keys; i++) {
    ground_truth[h_keys[i]] = h_values[i];
  }

  // Iterate
  timer.start();
  auto iter = hash_map.getIterator();
  double iter_time = timer.elapsedMs();

  bool passed = true;
  uint32_t count = 0;
  uint32_t mismatches = 0;

  while (iter.hasNext()) {
    auto pair = iter.next();
    count++;

    if (ground_truth.find(pair.key) == ground_truth.end()) {
      mismatches++;
    } else if (ground_truth[pair.key] != pair.value) {
      mismatches++;
    }
  }

  if (count != num_keys || mismatches > 0) {
    passed = false;
  }

  printTestResult("Large table iteration", passed);
  std::cout << "  Keys: " << num_keys << std::endl;
  std::cout << "  Entries iterated: " << count << std::endl;
  std::cout << "  Mismatches: " << mismatches << std::endl;
  std::cout << "  Insert time: " << std::fixed << std::setprecision(2) << insert_time << " ms" << std::endl;
  std::cout << "  Iterator creation time: " << std::fixed << std::setprecision(2) << iter_time << " ms" << std::endl;

  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
}

// ============================================================================
// Test 5: Iterator Reset Functionality
// ============================================================================
void test_iterator_reset() {
  printSectionHeader("Test 5: Iterator Reset Functionality");

  const uint32_t num_buckets = 500;
  const uint32_t num_keys = 10;
  GpuHashMap<uint32_t, uint32_t> hash_map(num_buckets, 0, 42, false);

  // Insert some keys
  std::vector<uint32_t> h_keys, h_values;
  generateSequentialKeys<uint32_t>(h_keys, num_keys, 50);
  generateSequentialKeys<uint32_t>(h_values, num_keys, 500);

  uint32_t* d_keys = allocateDeviceArray<uint32_t>(num_keys);
  uint32_t* d_values = allocateDeviceArray<uint32_t>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);
  hash_map.buildTable(d_keys, d_values, num_keys);

  // Get iterator
  auto iter = hash_map.getIterator();

  // First iteration
  uint32_t first_count = 0;
  while (iter.hasNext()) {
    iter.next();
    first_count++;
  }

  // Reset and iterate again
  iter.reset();
  uint32_t second_count = 0;
  while (iter.hasNext()) {
    iter.next();
    second_count++;
  }

  bool passed = (first_count == num_keys) && (second_count == num_keys) && (first_count == second_count);

  printTestResult("Iterator reset functionality", passed);
  std::cout << "  First iteration: " << first_count << " entries" << std::endl;
  std::cout << "  Second iteration: " << second_count << " entries" << std::endl;

  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
}

// ============================================================================
// Test 6: Multiple Iterations (Idempotence)
// ============================================================================
void test_multiple_iterations() {
  printSectionHeader("Test 6: Multiple Iterations (Idempotence)");

  const uint32_t num_buckets = 500;
  const uint32_t num_keys = 15;
  GpuHashMap<uint32_t, uint32_t> hash_map(num_buckets, 0, 42, false);

  // Insert keys
  std::vector<uint32_t> h_keys, h_values;
  generateSequentialKeys<uint32_t>(h_keys, num_keys, 300);
  generateSequentialKeys<uint32_t>(h_values, num_keys, 3000);

  uint32_t* d_keys = allocateDeviceArray<uint32_t>(num_keys);
  uint32_t* d_values = allocateDeviceArray<uint32_t>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);
  hash_map.buildTable(d_keys, d_values, num_keys);

  // Get two separate iterators
  auto iter1 = hash_map.getIterator();
  auto iter2 = hash_map.getIterator();

  // Collect all pairs from both iterators
  std::vector<std::pair<uint32_t, uint32_t>> pairs1, pairs2;

  while (iter1.hasNext()) {
    auto pair = iter1.next();
    pairs1.push_back({pair.key, pair.value});
  }

  while (iter2.hasNext()) {
    auto pair = iter2.next();
    pairs2.push_back({pair.key, pair.value});
  }

  // Sort both vectors for comparison (order may differ)
  std::sort(pairs1.begin(), pairs1.end());
  std::sort(pairs2.begin(), pairs2.end());

  bool passed = (pairs1.size() == num_keys) && (pairs2.size() == num_keys) && (pairs1 == pairs2);

  printTestResult("Multiple iterations produce same results", passed);
  std::cout << "  Iterator 1: " << pairs1.size() << " entries" << std::endl;
  std::cout << "  Iterator 2: " << pairs2.size() << " entries" << std::endl;

  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
}

// ============================================================================
// Test 7: High Load Factor
// ============================================================================
void test_high_load_factor() {
  printSectionHeader("Test 7: High Load Factor (0.9)");

  const uint32_t num_keys = 9000;
  const uint32_t num_buckets = 10000;  // Load factor 0.9
  GpuHashMap<uint32_t, uint32_t> hash_map(num_buckets, 0, 42, false);

  // Generate unique keys
  std::vector<uint32_t> h_keys, h_values;
  generateRandomUniqueKeys<uint32_t>(h_keys, num_keys, 456);
  generateSequentialKeys<uint32_t>(h_values, num_keys, 50000);

  uint32_t* d_keys = allocateDeviceArray<uint32_t>(num_keys);
  uint32_t* d_values = allocateDeviceArray<uint32_t>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);
  hash_map.buildTable(d_keys, d_values, num_keys);

  // Count via countTable
  uint32_t count_table_result = hash_map.countTable();

  // Count via iterator
  auto iter = hash_map.getIterator();
  uint32_t iter_count = 0;
  while (iter.hasNext()) {
    iter.next();
    iter_count++;
  }

  bool passed = (iter_count == count_table_result) && (iter.size() == count_table_result);

  printTestResult("High load factor iteration", passed);
  std::cout << "  Expected (via countTable): " << count_table_result << std::endl;
  std::cout << "  Iterator count: " << iter_count << std::endl;
  std::cout << "  Iterator size(): " << iter.size() << std::endl;
  std::cout << "  Load factor: " << std::fixed << std::setprecision(2)
            << (double)count_table_result / num_buckets << std::endl;

  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
}

// ============================================================================
// Test 8: Iterator After Clear
// ============================================================================
void test_iterator_after_clear() {
  printSectionHeader("Test 8: Iterator After Clear");

  const uint32_t num_buckets = 1000;
  const uint32_t num_keys = 50;
  GpuHashMap<uint32_t, uint32_t> hash_map(num_buckets, 0, 42, false);

  // Insert keys
  std::vector<uint32_t> h_keys, h_values;
  generateSequentialKeys<uint32_t>(h_keys, num_keys, 400);
  generateSequentialKeys<uint32_t>(h_values, num_keys, 4000);

  uint32_t* d_keys = allocateDeviceArray<uint32_t>(num_keys);
  uint32_t* d_values = allocateDeviceArray<uint32_t>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);
  hash_map.buildTable(d_keys, d_values, num_keys);

  // Clear the table
  hash_map.clear();

  // Get iterator after clear
  auto iter = hash_map.getIterator();

  bool passed = (iter.size() == 0) && (!iter.hasNext());

  printTestResult("Iterator after clear", passed);
  std::cout << "  Iterator size after clear: " << iter.size() << std::endl;

  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
}

// ============================================================================
// Main Function
// ============================================================================
int main() {
  std::cout << "\n";
  std::cout << COLOR_BLUE << "========================================" << COLOR_RESET << std::endl;
  std::cout << COLOR_BLUE << "  GPU Hash Map Iterator Test Suite" << COLOR_RESET << std::endl;
  std::cout << COLOR_BLUE << "========================================" << COLOR_RESET << std::endl;
  std::cout << "\n";

  test_empty_iteration();
  test_basic_iteration();
  test_iteration_after_deletions();
  test_large_table_iteration();
  test_iterator_reset();
  test_multiple_iterations();
  test_high_load_factor();
  test_iterator_after_clear();

  std::cout << "\n";
  std::cout << COLOR_BLUE << "========================================" << COLOR_RESET << std::endl;
  std::cout << COLOR_BLUE << "  All Iterator Tests Completed!" << COLOR_RESET << std::endl;
  std::cout << COLOR_BLUE << "========================================" << COLOR_RESET << std::endl;
  std::cout << "\n";

  return 0;
}
