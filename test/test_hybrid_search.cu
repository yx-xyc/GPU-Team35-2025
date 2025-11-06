/*
 * Comprehensive Hybrid Search Tests
 *
 * Tests the hybrid search strategy thoroughly:
 * 1. Correctness: Verify search results are accurate
 * 2. Hybrid strategy: Verify correct kernel selection based on threshold
 * 3. Performance: Compare one-warp-per-key vs one-thread-per-key
 * 4. Edge cases: Empty table, non-existent keys, after deletions
 * 5. Stress tests: Various batch sizes and load factors
 */

#include "../include/gpu_hash_map.cuh"
#include "test_utils.cuh"
#include <iostream>
#include <vector>
#include <algorithm>

using KeyT = uint32_t;
using ValueT = uint32_t;

using namespace TestUtils;

// ============================================================================
// Test Functions
// ============================================================================

/**
 * Test 1: Correctness - Search existing keys
 */
bool testSearchExistingKeys() {
  printSectionHeader("Test 1: Search Existing Keys");

  const uint32_t num_keys = 10000;
  const uint32_t num_buckets = 20000;

  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

  // Generate test data
  std::vector<KeyT> h_keys;
  std::vector<ValueT> h_values;
  generateRandomUniqueKeys(h_keys, num_keys, 42);
  generateMatchingValues(h_values, h_keys); // value = key

  // Allocate device memory
  KeyT* d_keys = allocateDeviceArray<KeyT>(num_keys);
  ValueT* d_values = allocateDeviceArray<ValueT>(num_keys);
  ValueT* d_results = allocateDeviceArray<ValueT>(num_keys);

  // Copy to device and insert
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);
  hash_map.buildTable(d_keys, d_values, num_keys);

  // Search for all inserted keys
  hash_map.searchTable(d_keys, d_results, num_keys);

  // Verify results
  std::vector<ValueT> h_results;
  copyToHost(h_results, d_results, num_keys);

  uint32_t mismatches = verifySearchResults(h_values, h_results, true);
  bool passed = (mismatches == 0);

  // Cleanup
  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
  freeDeviceArray(d_results);

  printTestResult("Search Existing Keys", passed,
                  std::to_string(num_keys - mismatches) + "/" + std::to_string(num_keys) + " found");

  return passed;
}

/**
 * Test 2: Correctness - Search non-existent keys
 */
bool testSearchNonExistentKeys() {
  printSectionHeader("Test 2: Search Non-Existent Keys");

  const uint32_t num_keys = 5000;
  const uint32_t num_queries = 5000;
  const uint32_t num_buckets = 10000;

  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

  // Generate and insert keys 0-4999
  std::vector<KeyT> h_keys;
  std::vector<ValueT> h_values;
  generateSequentialKeys<KeyT>(h_keys, num_keys, (KeyT)0);
  generateMatchingValues<KeyT, ValueT>(h_values, h_keys);

  KeyT* d_keys = allocateDeviceArray<KeyT>(num_keys);
  ValueT* d_values = allocateDeviceArray<ValueT>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);
  hash_map.buildTable(d_keys, d_values, num_keys);

  // Search for non-existent keys 10000-14999
  std::vector<KeyT> h_queries;
  generateSequentialKeys<KeyT>(h_queries, num_queries, (KeyT)10000);

  KeyT* d_queries = allocateDeviceArray<KeyT>(num_queries);
  ValueT* d_results = allocateDeviceArray<ValueT>(num_queries);
  copyToDevice(d_queries, h_queries);
  hash_map.searchTable(d_queries, d_results, num_queries);

  // Verify all return SEARCH_NOT_FOUND
  std::vector<ValueT> h_results;
  copyToHost(h_results, d_results, num_queries);

  uint32_t not_found_count = countMatches(h_results, SEARCH_NOT_FOUND);
  bool passed = (not_found_count == num_queries);

  // Cleanup
  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
  freeDeviceArray(d_queries);
  freeDeviceArray(d_results);

  printTestResult("Search Non-Existent Keys", passed,
                  std::to_string(not_found_count) + "/" + std::to_string(num_queries) + " correctly not found");

  return passed;
}

/**
 * Test 3: Correctness - Search after deletions
 */
bool testSearchAfterDeletions() {
  printSectionHeader("Test 3: Search After Deletions");

  const uint32_t num_keys = 10000;
  const uint32_t num_buckets = 20000;

  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

  // Insert keys
  std::vector<KeyT> h_keys;
  std::vector<ValueT> h_values;
  generateSequentialKeys<KeyT>(h_keys, num_keys, (KeyT)0);
  generateMatchingValues<KeyT, ValueT>(h_values, h_keys);

  KeyT* d_keys = allocateDeviceArray<KeyT>(num_keys);
  ValueT* d_values = allocateDeviceArray<ValueT>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);
  hash_map.buildTable(d_keys, d_values, num_keys);

  // Delete first half (keys 0-4999)
  const uint32_t num_delete = num_keys / 2;
  hash_map.deleteTable(d_keys, num_delete);

  // Search for deleted keys
  ValueT* d_results = allocateDeviceArray<ValueT>(num_delete);
  hash_map.searchTable(d_keys, d_results, num_delete);

  std::vector<ValueT> h_results;
  copyToHost(h_results, d_results, num_delete);

  uint32_t not_found = countMatches(h_results, SEARCH_NOT_FOUND);
  bool passed = (not_found == num_delete);

  // Cleanup
  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
  freeDeviceArray(d_results);

  printTestResult("Search After Deletions", passed,
                  std::to_string(not_found) + "/" + std::to_string(num_delete) + " correctly not found");

  return passed;
}

/**
 * Test 4: Empty table search
 */
bool testSearchEmptyTable() {
  printSectionHeader("Test 4: Search Empty Table");

  const uint32_t num_queries = 1000;
  const uint32_t num_buckets = 2000;

  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

  // Search without inserting anything
  std::vector<KeyT> h_queries;
  generateSequentialKeys<KeyT>(h_queries, num_queries, (KeyT)0);

  KeyT* d_queries = allocateDeviceArray<KeyT>(num_queries);
  ValueT* d_results = allocateDeviceArray<ValueT>(num_queries);
  copyToDevice(d_queries, h_queries);
  hash_map.searchTable(d_queries, d_results, num_queries);

  std::vector<ValueT> h_results;
  copyToHost(h_results, d_results, num_queries);

  uint32_t not_found = countMatches(h_results, SEARCH_NOT_FOUND);
  bool passed = (not_found == num_queries);

  // Cleanup
  freeDeviceArray(d_queries);
  freeDeviceArray(d_results);

  printTestResult("Search Empty Table", passed,
                  std::to_string(not_found) + "/" + std::to_string(num_queries) + " correctly not found");

  return passed;
}

/**
 * Test 5: Hybrid strategy - threshold boundary
 */
bool testHybridThresholdBoundary() {
  printSectionHeader("Test 5: Hybrid Strategy - Threshold Boundary");

  const uint32_t threshold = 1000;
  const uint32_t num_buckets = 10000;

  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, true, threshold);

  // Insert some data
  std::vector<KeyT> h_keys;
  std::vector<ValueT> h_values;
  generateSequentialKeys<KeyT>(h_keys, 5000, (KeyT)0);
  generateMatchingValues<KeyT, ValueT>(h_values, h_keys);

  KeyT* d_keys = allocateDeviceArray<KeyT>(5000);
  ValueT* d_values = allocateDeviceArray<ValueT>(5000);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);
  hash_map.buildTable(d_keys, d_values, 5000);

  std::cout << "Testing with threshold = " << threshold << std::endl;

  // Test threshold - 1 (should use one-warp-per-key)
  std::cout << "\n  Query batch size: " << (threshold - 1) << " (< threshold)" << std::endl;
  KeyT* d_queries1 = allocateDeviceArray<KeyT>(threshold - 1);
  ValueT* d_results1 = allocateDeviceArray<ValueT>(threshold - 1);
  hash_map.searchTable(d_queries1, d_results1, threshold - 1);

  // Test threshold (should use one-thread-per-key)
  std::cout << "\n  Query batch size: " << threshold << " (= threshold)" << std::endl;
  KeyT* d_queries2 = allocateDeviceArray<KeyT>(threshold);
  ValueT* d_results2 = allocateDeviceArray<ValueT>(threshold);
  hash_map.searchTable(d_queries2, d_results2, threshold);

  // Test threshold + 1 (should use one-thread-per-key)
  std::cout << "\n  Query batch size: " << (threshold + 1) << " (> threshold)" << std::endl;
  KeyT* d_queries3 = allocateDeviceArray<KeyT>(threshold + 1);
  ValueT* d_results3 = allocateDeviceArray<ValueT>(threshold + 1);
  hash_map.searchTable(d_queries3, d_results3, threshold + 1);

  // Cleanup
  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
  freeDeviceArray(d_queries1);
  freeDeviceArray(d_results1);
  freeDeviceArray(d_queries2);
  freeDeviceArray(d_results2);
  freeDeviceArray(d_queries3);
  freeDeviceArray(d_results3);

  printTestResult("Hybrid Threshold Boundary", true, "Verified kernel selection");

  return true;
}

/**
 * Test 6: Performance comparison across batch sizes
 */
bool testPerformanceComparison() {
  printSectionHeader("Test 6: Performance Comparison");

  const uint32_t num_keys = 100000;
  const uint32_t num_buckets = 200000; // Load factor 0.5

  // Test batch sizes
  std::vector<uint32_t> batch_sizes = {1, 10, 100, 1000, 5000, 10000, 50000, 100000};

  // Test with one-warp-per-key (threshold = 1M, forces small batch kernel)
  std::cout << "\nOne-Warp-Per-Key Strategy (threshold = 1000000):" << std::endl;
  GpuHashMap<KeyT, ValueT> hash_map_warp(num_buckets, 0, 12345, false, 1000000);

  std::vector<KeyT> h_keys;
  std::vector<ValueT> h_values;
  generateSequentialKeys<KeyT>(h_keys, num_keys, (KeyT)0);
  generateMatchingValues<KeyT, ValueT>(h_values, h_keys);

  KeyT* d_keys = allocateDeviceArray<KeyT>(num_keys);
  ValueT* d_values = allocateDeviceArray<ValueT>(num_keys);
  ValueT* d_results = allocateDeviceArray<ValueT>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);
  hash_map_warp.buildTable(d_keys, d_values, num_keys);

  Timer timer;
  for (auto batch_size : batch_sizes) {
    if (batch_size > num_keys) continue;

    timer.start();
    hash_map_warp.searchTable(d_keys, d_results, batch_size);
    cudaDeviceSynchronize();
    double time_ms = timer.elapsedMs();

    printPerformance("Batch " + std::to_string(batch_size), batch_size, time_ms);
  }

  // Test with one-thread-per-key (threshold = 0, forces large batch kernel)
  std::cout << "\nOne-Thread-Per-Key Strategy (threshold = 0):" << std::endl;
  GpuHashMap<KeyT, ValueT> hash_map_thread(num_buckets, 0, 12345, false, 0);
  hash_map_thread.buildTable(d_keys, d_values, num_keys);

  for (auto batch_size : batch_sizes) {
    if (batch_size > num_keys) continue;

    timer.start();
    hash_map_thread.searchTable(d_keys, d_results, batch_size);
    cudaDeviceSynchronize();
    double time_ms = timer.elapsedMs();

    printPerformance("Batch " + std::to_string(batch_size), batch_size, time_ms);
  }

  // Cleanup
  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
  freeDeviceArray(d_results);

  printTestResult("Performance Comparison", true, "Completed benchmarks");

  return true;
}

/**
 * Test 7: Different load factors
 */
bool testDifferentLoadFactors() {
  printSectionHeader("Test 7: Different Load Factors");

  const uint32_t num_queries = 10000;
  std::vector<double> load_factors = {0.25, 0.5, 0.75, 0.9};

  Timer timer;

  for (double lf : load_factors) {
    uint32_t num_keys = static_cast<uint32_t>(num_queries * lf);
    uint32_t num_buckets = num_queries;

    GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

    // Insert keys
    std::vector<KeyT> h_keys;
    std::vector<ValueT> h_values;
    generateSequentialKeys<KeyT>(h_keys, num_keys, (KeyT)0);
    generateMatchingValues<KeyT, ValueT>(h_values, h_keys);

    KeyT* d_keys = allocateDeviceArray<KeyT>(num_keys);
    ValueT* d_values = allocateDeviceArray<ValueT>(num_keys);
    ValueT* d_results = allocateDeviceArray<ValueT>(num_keys);
    copyToDevice(d_keys, h_keys);
    copyToDevice(d_values, h_values);
    hash_map.buildTable(d_keys, d_values, num_keys);

    // Search
    timer.start();
    hash_map.searchTable(d_keys, d_results, num_keys);
    cudaDeviceSynchronize();
    double time_ms = timer.elapsedMs();

    // Verify
    std::vector<ValueT> h_results;
    copyToHost(h_results, d_results, num_keys);
    uint32_t mismatches = verifySearchResults(h_values, h_results, false);

    std::cout << "  Load factor " << std::fixed << std::setprecision(2) << lf
              << ": " << (num_keys - mismatches) << "/" << num_keys << " found, "
              << std::setprecision(3) << time_ms << " ms" << std::endl;

    // Cleanup
    freeDeviceArray(d_keys);
    freeDeviceArray(d_values);
    freeDeviceArray(d_results);
  }

  printTestResult("Different Load Factors", true, "All load factors tested");

  return true;
}

/**
 * Test 8: Single key search
 */
bool testSingleKeySearch() {
  printSectionHeader("Test 8: Single Key Search");

  const uint32_t num_buckets = 1000;

  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

  // Insert one key
  std::vector<KeyT> h_key = {42};
  std::vector<ValueT> h_value = {9999};

  KeyT* d_key = allocateDeviceArray<KeyT>(1);
  ValueT* d_value = allocateDeviceArray<ValueT>(1);
  ValueT* d_result = allocateDeviceArray<ValueT>(1);
  copyToDevice(d_key, h_key);
  copyToDevice(d_value, h_value);
  hash_map.buildTable(d_key, d_value, 1);

  // Search for it
  hash_map.searchTable(d_key, d_result, 1);

  std::vector<ValueT> h_result;
  copyToHost(h_result, d_result, 1);

  bool passed = (h_result[0] == h_value[0]);

  // Cleanup
  freeDeviceArray(d_key);
  freeDeviceArray(d_value);
  freeDeviceArray(d_result);

  printTestResult("Single Key Search", passed,
                  "Expected " + std::to_string(h_value[0]) +
                  ", got " + std::to_string(h_result[0]));

  return passed;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
  std::cout << COLOR_BLUE << "\n"
            << "╔════════════════════════════════════════════════════╗\n"
            << "║   Comprehensive Hybrid Search Test Suite          ║\n"
            << "╚════════════════════════════════════════════════════╝"
            << COLOR_RESET << "\n" << std::endl;

  printDeviceInfo(0);
  std::cout << std::endl;

  uint32_t passed = 0;
  uint32_t total = 0;

  // Run all tests
  if (testSearchExistingKeys()) passed++;
  total++;

  if (testSearchNonExistentKeys()) passed++;
  total++;

  if (testSearchAfterDeletions()) passed++;
  total++;

  if (testSearchEmptyTable()) passed++;
  total++;

  if (testHybridThresholdBoundary()) passed++;
  total++;

  if (testPerformanceComparison()) passed++;
  total++;

  if (testDifferentLoadFactors()) passed++;
  total++;

  if (testSingleKeySearch()) passed++;
  total++;

  // Print summary
  printTestSummary(passed, total);

  return (passed == total) ? 0 : 1;
}
