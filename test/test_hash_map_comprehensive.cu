/*
 * Comprehensive GPU Hash Map Tests
 *
 * Tests all hash map operations end-to-end:
 * 1. Insert (buildTable) - single, bulk, duplicates
 * 2. Search (searchTable) - found, not found
 * 3. Delete (deleteTable) - existing, non-existent
 * 4. Count (countTable) - empty, after ops
 * 5. Stress tests - various load factors, table sizes
 * 6. Edge cases - full table, repeated ops, TOMBSTONE reuse
 */

#include "../include/gpu_hash_map.cuh"
#include "test_utils.cuh"
#include <iostream>
#include <vector>

using KeyT = uint32_t;
using ValueT = uint32_t;

using namespace TestUtils;

// ============================================================================
// Basic Operation Tests
// ============================================================================

/**
 * Test 1: Insert and Count
 */
bool testInsertAndCount() {
  printSectionHeader("Test 1: Insert and Count");

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

  // Count
  uint32_t count = hash_map.countTable();

  bool passed = (count == num_keys);

  // Cleanup
  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);

  printTestResult("Insert and Count", passed,
                  "Count: " + std::to_string(count) + "/" + std::to_string(num_keys));

  return passed;
}

/**
 * Test 2: Insert, Search, Verify
 */
bool testInsertSearchVerify() {
  printSectionHeader("Test 2: Insert, Search, Verify");

  const uint32_t num_keys = 50000;
  const uint32_t num_buckets = 100000;

  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

  // Insert keys
  std::vector<KeyT> h_keys;
  std::vector<ValueT> h_values;
  generateRandomUniqueKeys(h_keys, num_keys, 42);
  generateMatchingValues<KeyT, ValueT>(h_values, h_keys);

  KeyT* d_keys = allocateDeviceArray<KeyT>(num_keys);
  ValueT* d_values = allocateDeviceArray<ValueT>(num_keys);
  ValueT* d_results = allocateDeviceArray<ValueT>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);

  hash_map.buildTable(d_keys, d_values, num_keys);

  // Search
  hash_map.searchTable(d_keys, d_results, num_keys);

  // Verify
  std::vector<ValueT> h_results;
  copyToHost(h_results, d_results, num_keys);

  uint32_t mismatches = verifySearchResults(h_values, h_results, false);
  bool passed = (mismatches == 0);

  // Cleanup
  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
  freeDeviceArray(d_results);

  printTestResult("Insert, Search, Verify", passed,
                  std::to_string(num_keys - mismatches) + "/" + std::to_string(num_keys) + " correct");

  return passed;
}

/**
 * Test 3: Insert Duplicates (should update values)
 */
bool testInsertDuplicates() {
  printSectionHeader("Test 3: Insert Duplicates");

  const uint32_t num_keys = 1000;
  const uint32_t num_buckets = 2000;

  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

  // Insert keys with value = key
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

  // Count should be num_keys
  uint32_t count1 = hash_map.countTable();

  // Re-insert same keys with value = key + 1000
  std::vector<ValueT> h_values_new;
  generateMatchingValues(h_values_new, h_keys);
  for (auto& val : h_values_new) val += 1000;
  copyToDevice(d_values, h_values_new);

  hash_map.buildTable(d_keys, d_values, num_keys);

  // Count should still be num_keys (no new keys added)
  uint32_t count2 = hash_map.countTable();

  // Search should return updated values
  hash_map.searchTable(d_keys, d_results, num_keys);

  std::vector<ValueT> h_results;
  copyToHost(h_results, d_results, num_keys);

  uint32_t mismatches = verifySearchResults(h_values_new, h_results, false);

  bool passed = (count1 == num_keys) && (count2 == num_keys) && (mismatches == 0);

  // Cleanup
  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
  freeDeviceArray(d_results);

  printTestResult("Insert Duplicates", passed,
                  "Count: " + std::to_string(count1) + " → " + std::to_string(count2) +
                  ", Values updated: " + std::to_string(num_keys - mismatches) + "/" + std::to_string(num_keys));

  return passed;
}

/**
 * Test 4: Delete and Verify
 */
bool testDeleteAndVerify() {
  printSectionHeader("Test 4: Delete and Verify");

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
  ValueT* d_results = allocateDeviceArray<ValueT>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);

  hash_map.buildTable(d_keys, d_values, num_keys);

  // Count before delete
  uint32_t count_before = hash_map.countTable();

  // Delete half
  const uint32_t num_delete = num_keys / 2;
  hash_map.deleteTable(d_keys, num_delete);

  // Count after delete (should be reduced)
  uint32_t count_after = hash_map.countTable();

  // Search deleted keys (should not be found)
  hash_map.searchTable(d_keys, d_results, num_delete);
  std::vector<ValueT> h_results_deleted;
  copyToHost(h_results_deleted, d_results, num_delete);
  uint32_t deleted_not_found = countMatches(h_results_deleted, SEARCH_NOT_FOUND);

  // Search remaining keys (should be found)
  KeyT* d_remaining = d_keys + num_delete;
  ValueT* d_results_remaining = d_results + num_delete;
  uint32_t num_remaining = num_keys - num_delete;
  hash_map.searchTable(d_remaining, d_results_remaining, num_remaining);
  std::vector<ValueT> h_results_remaining;
  copyToHost(h_results_remaining, d_results_remaining, num_remaining);

  std::vector<ValueT> h_expected_remaining(h_values.begin() + num_delete, h_values.end());
  uint32_t remaining_matches = num_remaining - verifySearchResults(h_expected_remaining, h_results_remaining, false);

  bool passed = (count_before == num_keys) &&
                (count_after == num_remaining) &&
                (deleted_not_found == num_delete) &&
                (remaining_matches == num_remaining);

  // Cleanup
  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
  freeDeviceArray(d_results);

  printTestResult("Delete and Verify", passed,
                  "Count: " + std::to_string(count_before) + " → " + std::to_string(count_after) +
                  ", Deleted not found: " + std::to_string(deleted_not_found) + "/" + std::to_string(num_delete) +
                  ", Remaining found: " + std::to_string(remaining_matches) + "/" + std::to_string(num_remaining));

  return passed;
}

/**
 * Test 5: Delete Non-Existent Keys (should succeed silently)
 */
bool testDeleteNonExistent() {
  printSectionHeader("Test 5: Delete Non-Existent Keys");

  const uint32_t num_buckets = 1000;

  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

  // Try to delete keys without inserting (should not crash)
  std::vector<KeyT> h_keys;
  generateSequentialKeys<KeyT>(h_keys, 100, (KeyT)0);

  KeyT* d_keys = allocateDeviceArray<KeyT>(100);
  copyToDevice(d_keys, h_keys);

  hash_map.deleteTable(d_keys, 100);

  // Count should be 0
  uint32_t count = hash_map.countTable();

  bool passed = (count == 0);

  // Cleanup
  freeDeviceArray(d_keys);

  printTestResult("Delete Non-Existent", passed, "Silent success, count remains 0");

  return passed;
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/**
 * Test 6: Empty Table Operations
 */
bool testEmptyTable() {
  printSectionHeader("Test 6: Empty Table Operations");

  const uint32_t num_buckets = 1000;

  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

  // Count empty table
  uint32_t count1 = hash_map.countTable();

  // Search empty table
  std::vector<KeyT> h_queries;
  generateSequentialKeys<KeyT>(h_queries, 10, (KeyT)0);

  KeyT* d_queries = allocateDeviceArray<KeyT>(10);
  ValueT* d_results = allocateDeviceArray<ValueT>(10);
  copyToDevice(d_queries, h_queries);

  hash_map.searchTable(d_queries, d_results, 10);

  std::vector<ValueT> h_results;
  copyToHost(h_results, d_results, 10);

  uint32_t not_found = countMatches(h_results, SEARCH_NOT_FOUND);

  // Delete from empty table
  hash_map.deleteTable(d_queries, 10);

  uint32_t count2 = hash_map.countTable();

  bool passed = (count1 == 0) && (count2 == 0) && (not_found == 10);

  // Cleanup
  freeDeviceArray(d_queries);
  freeDeviceArray(d_results);

  printTestResult("Empty Table Operations", passed, "All operations safe on empty table");

  return passed;
}

/**
 * Test 7: Repeated Insert-Delete-Insert (TOMBSTONE reuse)
 */
bool testRepeatedInsertDelete() {
  printSectionHeader("Test 7: Repeated Insert-Delete-Insert");

  const uint32_t num_keys = 1000;
  const uint32_t num_buckets = 2000;

  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

  std::vector<KeyT> h_keys;
  std::vector<ValueT> h_values;
  generateSequentialKeys<KeyT>(h_keys, num_keys, (KeyT)0);
  generateMatchingValues<KeyT, ValueT>(h_values, h_keys);

  KeyT* d_keys = allocateDeviceArray<KeyT>(num_keys);
  ValueT* d_values = allocateDeviceArray<ValueT>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);

  // Cycle 1: Insert
  hash_map.buildTable(d_keys, d_values, num_keys);
  uint32_t count1 = hash_map.countTable();

  // Cycle 1: Delete
  hash_map.deleteTable(d_keys, num_keys);
  uint32_t count2 = hash_map.countTable();

  // Cycle 2: Insert again (should reuse TOMBSTONE slots)
  hash_map.buildTable(d_keys, d_values, num_keys);
  uint32_t count3 = hash_map.countTable();

  // Cycle 2: Delete
  hash_map.deleteTable(d_keys, num_keys);
  uint32_t count4 = hash_map.countTable();

  // Cycle 3: Insert again
  hash_map.buildTable(d_keys, d_values, num_keys);
  uint32_t count5 = hash_map.countTable();

  bool passed = (count1 == num_keys) && (count2 == 0) &&
                (count3 == num_keys) && (count4 == 0) &&
                (count5 == num_keys);

  // Cleanup
  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);

  printTestResult("Repeated Insert-Delete-Insert", passed,
                  "Counts: " + std::to_string(count1) + " → " + std::to_string(count2) +
                  " → " + std::to_string(count3) + " → " + std::to_string(count4) +
                  " → " + std::to_string(count5));

  return passed;
}

// ============================================================================
// Stress Tests
// ============================================================================

/**
 * Test 8: Various Load Factors
 */
bool testVariousLoadFactors() {
  printSectionHeader("Test 8: Various Load Factors");

  const uint32_t num_buckets = 10000;
  std::vector<double> load_factors = {0.1, 0.25, 0.5, 0.75, 0.8, 0.9};

  bool all_passed = true;

  for (double lf : load_factors) {
    uint32_t num_keys = static_cast<uint32_t>(num_buckets * lf);

    GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

    std::vector<KeyT> h_keys;
    std::vector<ValueT> h_values;
    generateSequentialKeys<KeyT>(h_keys, num_keys, (KeyT)0);
    generateMatchingValues<KeyT, ValueT>(h_values, h_keys);

    KeyT* d_keys = allocateDeviceArray<KeyT>(num_keys);
    ValueT* d_values = allocateDeviceArray<ValueT>(num_keys);
    ValueT* d_results = allocateDeviceArray<ValueT>(num_keys);
    copyToDevice(d_keys, h_keys);
    copyToDevice(d_values, h_values);

    // Insert
    Timer timer;
    timer.start();
    hash_map.buildTable(d_keys, d_values, num_keys);
    cudaDeviceSynchronize();
    double insert_time = timer.elapsedMs();

    // Count
    uint32_t count = hash_map.countTable();

    // Search
    timer.start();
    hash_map.searchTable(d_keys, d_results, num_keys);
    cudaDeviceSynchronize();
    double search_time = timer.elapsedMs();

    // Verify
    std::vector<ValueT> h_results;
    copyToHost(h_results, d_results, num_keys);
    uint32_t mismatches = verifySearchResults(h_values, h_results, false);

    bool passed = (count == num_keys) && (mismatches == 0);
    all_passed &= passed;

    std::cout << "  Load factor " << std::fixed << std::setprecision(2) << lf
              << ": Insert " << std::setprecision(3) << insert_time << " ms, "
              << "Search " << search_time << " ms, "
              << "Count " << count << "/" << num_keys
              << (passed ? " ✓" : " ✗") << std::endl;

    // Cleanup
    freeDeviceArray(d_keys);
    freeDeviceArray(d_values);
    freeDeviceArray(d_results);
  }

  printTestResult("Various Load Factors", all_passed, "All load factors tested");

  return all_passed;
}

/**
 * Test 9: Different Table Sizes
 */
bool testDifferentTableSizes() {
  printSectionHeader("Test 9: Different Table Sizes");

  std::vector<uint32_t> sizes = {100, 1000, 10000, 100000};
  const double load_factor = 0.5;

  bool all_passed = true;

  for (uint32_t num_buckets : sizes) {
    uint32_t num_keys = static_cast<uint32_t>(num_buckets * load_factor);

    GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

    std::vector<KeyT> h_keys;
    std::vector<ValueT> h_values;
    generateRandomUniqueKeys(h_keys, num_keys, 42);
    generateMatchingValues<KeyT, ValueT>(h_values, h_keys);

    KeyT* d_keys = allocateDeviceArray<KeyT>(num_keys);
    ValueT* d_values = allocateDeviceArray<ValueT>(num_keys);
    ValueT* d_results = allocateDeviceArray<ValueT>(num_keys);
    copyToDevice(d_keys, h_keys);
    copyToDevice(d_values, h_values);

    // Insert and search
    hash_map.buildTable(d_keys, d_values, num_keys);
    hash_map.searchTable(d_keys, d_results, num_keys);

    // Verify
    std::vector<ValueT> h_results;
    copyToHost(h_results, d_results, num_keys);
    uint32_t mismatches = verifySearchResults(h_values, h_results, false);

    uint32_t count = hash_map.countTable();

    bool passed = (count == num_keys) && (mismatches == 0);
    all_passed &= passed;

    std::cout << "  Buckets: " << std::setw(7) << num_buckets
              << ", Keys: " << std::setw(7) << num_keys
              << ", Count: " << count
              << ", Matches: " << (num_keys - mismatches) << "/" << num_keys
              << (passed ? " ✓" : " ✗") << std::endl;

    // Cleanup
    freeDeviceArray(d_keys);
    freeDeviceArray(d_values);
    freeDeviceArray(d_results);
  }

  printTestResult("Different Table Sizes", all_passed, "All sizes tested");

  return all_passed;
}

/**
 * Test 10: Large Scale End-to-End
 */
bool testLargeScaleEndToEnd() {
  printSectionHeader("Test 10: Large Scale End-to-End");

  const uint32_t num_keys = 1000000;  // 1M keys
  const uint32_t num_buckets = 2000000; // Load factor 0.5

  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);

  std::cout << "  Generating " << num_keys << " random keys..." << std::endl;
  std::vector<KeyT> h_keys;
  std::vector<ValueT> h_values;
  generateRandomUniqueKeys(h_keys, num_keys, 42);
  generateMatchingValues<KeyT, ValueT>(h_values, h_keys);

  KeyT* d_keys = allocateDeviceArray<KeyT>(num_keys);
  ValueT* d_values = allocateDeviceArray<ValueT>(num_keys);
  ValueT* d_results = allocateDeviceArray<ValueT>(num_keys);
  copyToDevice(d_keys, h_keys);
  copyToDevice(d_values, h_values);

  // Insert
  std::cout << "  Inserting..." << std::endl;
  Timer timer;
  timer.start();
  hash_map.buildTable(d_keys, d_values, num_keys);
  cudaDeviceSynchronize();
  double insert_time = timer.elapsedMs();
  printPerformance("Insert", num_keys, insert_time);

  // Count
  uint32_t count = hash_map.countTable();
  std::cout << "  Count: " << count << " / " << num_keys << std::endl;

  // Search
  std::cout << "  Searching..." << std::endl;
  timer.start();
  hash_map.searchTable(d_keys, d_results, num_keys);
  cudaDeviceSynchronize();
  double search_time = timer.elapsedMs();
  printPerformance("Search", num_keys, search_time);

  // Verify
  std::cout << "  Verifying results..." << std::endl;
  std::vector<ValueT> h_results;
  copyToHost(h_results, d_results, num_keys);
  uint32_t mismatches = verifySearchResults(h_values, h_results, false);

  // Delete half
  std::cout << "  Deleting 50%..." << std::endl;
  const uint32_t num_delete = num_keys / 2;
  timer.start();
  hash_map.deleteTable(d_keys, num_delete);
  cudaDeviceSynchronize();
  double delete_time = timer.elapsedMs();
  printPerformance("Delete", num_delete, delete_time);

  uint32_t count_after_delete = hash_map.countTable();
  std::cout << "  Count after delete: " << count_after_delete << " (expected ~" << (num_keys - num_delete) << ")" << std::endl;

  bool passed = (count == num_keys) &&
                (mismatches == 0) &&
                (count_after_delete >= (num_keys - num_delete) * 0.99); // Allow 1% tolerance

  // Cleanup
  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
  freeDeviceArray(d_results);

  printTestResult("Large Scale End-to-End", passed, "1M keys processed successfully");

  return passed;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
  std::cout << COLOR_BLUE << "\n"
            << "╔════════════════════════════════════════════════════╗\n"
            << "║    Comprehensive GPU Hash Map Test Suite          ║\n"
            << "╚════════════════════════════════════════════════════╝"
            << COLOR_RESET << "\n" << std::endl;

  printDeviceInfo(0);
  std::cout << std::endl;

  uint32_t passed = 0;
  uint32_t total = 0;

  // Basic operation tests
  if (testInsertAndCount()) passed++;
  total++;

  if (testInsertSearchVerify()) passed++;
  total++;

  if (testInsertDuplicates()) passed++;
  total++;

  if (testDeleteAndVerify()) passed++;
  total++;

  if (testDeleteNonExistent()) passed++;
  total++;

  // Edge case tests
  if (testEmptyTable()) passed++;
  total++;

  if (testRepeatedInsertDelete()) passed++;
  total++;

  // Stress tests
  if (testVariousLoadFactors()) passed++;
  total++;

  if (testDifferentTableSizes()) passed++;
  total++;

  if (testLargeScaleEndToEnd()) passed++;
  total++;

  // Print summary
  printTestSummary(passed, total);

  return (passed == total) ? 0 : 1;
}
