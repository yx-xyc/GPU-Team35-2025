/*
 * Search Performance Experiment
 *
 * Tests search performance across different load factors and batch sizes
 * to understand the impact on the hybrid search strategy.
 *
 * Configuration:
 *   - Table size: 1M keys baseline (2M buckets at LF=0.5)
 *   - Load factors: 0.25, 0.5, 0.75, 0.9
 *   - Batch sizes: 100, 1K, 10K, 100K
 *   - Query pattern: 50/50 mixed (half hits, half misses)
 *   - Total: 16 configurations
 */

#include "gpu_hash_map.cuh"
#include "test_utils.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>

using KeyT = uint32_t;
using ValueT = uint32_t;

// ============================================================================
// CSV Output Functions
// ============================================================================

void writeCSVHeader(std::ofstream& csv) {
  csv << "load_factor,num_keys,num_buckets,batch_size,"
      << "warp_time_ms,warp_throughput_mops,"
      << "thread_time_ms,thread_throughput_mops,"
      << "speedup,faster_strategy\n";
}

void appendCSVRow(std::ofstream& csv,
                  double load_factor,
                  uint32_t num_keys,
                  uint32_t num_buckets,
                  uint32_t batch_size,
                  double warp_time_ms,
                  double warp_throughput,
                  double thread_time_ms,
                  double thread_throughput) {

  // Calculate speedup as faster/slower (always > 1.0)
  double speedup;
  std::string faster;
  if (warp_time_ms < thread_time_ms) {
    speedup = thread_time_ms / warp_time_ms;
    faster = "warp";
  } else {
    speedup = warp_time_ms / thread_time_ms;
    faster = "thread";
  }

  csv << std::fixed << std::setprecision(2) << load_factor << ","
      << num_keys << ","
      << num_buckets << ","
      << batch_size << ","
      << std::setprecision(4) << warp_time_ms << ","
      << std::setprecision(2) << warp_throughput << ","
      << std::setprecision(4) << thread_time_ms << ","
      << std::setprecision(2) << thread_throughput << ","
      << std::setprecision(2) << speedup << ","
      << faster << "\n";
  csv.flush();  // Ensure data is written immediately
}

// ============================================================================
// Test Configuration Structure
// ============================================================================

struct SearchResult {
  double warp_time_ms;
  double warp_throughput_mops;
  double thread_time_ms;
  double thread_throughput_mops;
  bool correctness_passed;
};

// ============================================================================
// Core Test Function
// ============================================================================

SearchResult runSearchTest(uint32_t num_keys,
                           uint32_t num_buckets,
                           uint32_t batch_size,
                           double load_factor) {
  SearchResult result;

  // Generate keys to insert (up to num_keys based on load factor)
  std::vector<KeyT> h_insert_keys(num_keys);
  std::vector<ValueT> h_insert_values(num_keys);
  TestUtils::generateSequentialKeys(h_insert_keys, num_keys, KeyT(1));  // Keys: 1, 2, 3, ...
  TestUtils::generateMatchingValues(h_insert_values, h_insert_keys);

  // Allocate device memory for insertions
  KeyT* d_insert_keys = TestUtils::allocateDeviceArray<KeyT>(num_keys);
  ValueT* d_insert_values = TestUtils::allocateDeviceArray<ValueT>(num_keys);

  if (!d_insert_keys || !d_insert_values) {
    result.warp_time_ms = -1.0;
    result.warp_throughput_mops = 0.0;
    result.thread_time_ms = -1.0;
    result.thread_throughput_mops = 0.0;
    result.correctness_passed = false;
    return result;
  }

  TestUtils::copyToDevice(d_insert_keys, h_insert_keys);
  TestUtils::copyToDevice(d_insert_values, h_insert_values);

  // Generate query batch: 50/50 mixed (half hits, half misses)
  std::vector<KeyT> h_query_keys(batch_size);
  std::vector<ValueT> h_expected_values(batch_size);

  uint32_t num_hits = batch_size / 2;
  uint32_t num_misses = batch_size - num_hits;

  // First half: keys that exist (1 to num_hits)
  for (uint32_t i = 0; i < num_hits; i++) {
    h_query_keys[i] = i + 1;
    h_expected_values[i] = i + 1;  // value = key
  }

  // Second half: keys that don't exist (num_keys+1 onwards)
  for (uint32_t i = 0; i < num_misses; i++) {
    h_query_keys[num_hits + i] = num_keys + 1 + i;
    h_expected_values[num_hits + i] = SEARCH_NOT_FOUND;
  }

  // Allocate device memory for queries
  KeyT* d_query_keys = TestUtils::allocateDeviceArray<KeyT>(batch_size);
  ValueT* d_query_results = TestUtils::allocateDeviceArray<ValueT>(batch_size);

  if (!d_query_keys || !d_query_results) {
    TestUtils::freeDeviceArray(d_insert_keys);
    TestUtils::freeDeviceArray(d_insert_values);
    result.warp_time_ms = -1.0;
    result.warp_throughput_mops = 0.0;
    result.thread_time_ms = -1.0;
    result.thread_throughput_mops = 0.0;
    result.correctness_passed = false;
    return result;
  }

  TestUtils::copyToDevice(d_query_keys, h_query_keys);

  const int num_runs = 20;
  TestUtils::Timer timer;

  // Create a single hash map
  GpuHashMap<KeyT, ValueT> hash_map(num_buckets, 0, 12345, false);
  hash_map.buildTable(d_insert_keys, d_insert_values, num_keys);

  // ========== Test 1: WARP-PER-KEY Strategy (now with stride loop!) ==========
  {
    // Warm-up
    hash_map.searchTableWarpPerKey(d_query_keys, d_query_results, batch_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Timed runs
    double total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
      timer.start();
      hash_map.searchTableWarpPerKey(d_query_keys, d_query_results, batch_size);
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());
      total_time += timer.elapsedMs();
    }

    result.warp_time_ms = total_time / num_runs;
    result.warp_throughput_mops = TestUtils::calculateThroughput(batch_size, result.warp_time_ms);

    // Verify correctness
    std::vector<ValueT> h_query_results;
    TestUtils::copyToHost(h_query_results, d_query_results, batch_size);
    uint32_t mismatches = TestUtils::verifySearchResults(h_expected_values, h_query_results, false);
    result.correctness_passed = (mismatches == 0);
  }

  // ========== Test 2: THREAD-PER-KEY Strategy ==========
  {
    // Warm-up
    hash_map.searchTableThreadPerKey(d_query_keys, d_query_results, batch_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Timed runs
    double total_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
      timer.start();
      hash_map.searchTableThreadPerKey(d_query_keys, d_query_results, batch_size);
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());
      total_time += timer.elapsedMs();
    }

    result.thread_time_ms = total_time / num_runs;
    result.thread_throughput_mops = TestUtils::calculateThroughput(batch_size, result.thread_time_ms);

    // Verify correctness
    std::vector<ValueT> h_query_results;
    TestUtils::copyToHost(h_query_results, d_query_results, batch_size);
    uint32_t mismatches = TestUtils::verifySearchResults(h_expected_values, h_query_results, false);
    result.correctness_passed = result.correctness_passed && (mismatches == 0);
  }

  // Cleanup
  TestUtils::freeDeviceArray(d_insert_keys);
  TestUtils::freeDeviceArray(d_insert_values);
  TestUtils::freeDeviceArray(d_query_keys);
  TestUtils::freeDeviceArray(d_query_results);

  return result;
}

// ============================================================================
// Main Experiment
// ============================================================================

void runExperiment(std::ofstream& csv) {
  const uint32_t base_buckets = 2000000;  // 2M buckets

  // Test configurations - focused on showing strategy crossover
  // Load factors from sparse (0.25) to extreme (0.99)
  const std::vector<double> load_factors = {0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99};

  // Granular batch sizes to clearly show warp vs thread crossover point
  // Very small (1-10): warp should win
  // Transition (20-100): competition zone
  // Large (500+): thread should dominate
  const std::vector<uint32_t> batch_sizes = {1, 5, 10, 20, 50, 100, 500, 1000, 10000, 100000};

  int config_num = 0;
  int total_configs = load_factors.size() * batch_sizes.size();

  std::cout << "\n" << COLOR_BLUE << "Starting Search Performance Experiment"
            << COLOR_RESET << std::endl;
  std::cout << "Total configurations: " << total_configs << std::endl;
  std::cout << "Base buckets: " << base_buckets << std::endl;
  std::cout << "Testing TWO strategies per configuration:" << std::endl;
  std::cout << "  1. Warp-per-key (NOW with stride loop optimization!)" << std::endl;
  std::cout << "  2. Thread-per-key" << std::endl;
  std::cout << "Load factors: 0.5 → 0.99 (moderate to extreme)" << std::endl;
  std::cout << "Batch sizes: 1, 5, 10, 20, 50, 100, 500, 1K, 10K, 100K" << std::endl;
  std::cout << "Query pattern: 50/50 mixed (hits/misses)\n" << std::endl;

  // Progress tracking
  int passed = 0;
  int failed = 0;

  for (double lf : load_factors) {
    uint32_t num_keys = static_cast<uint32_t>(std::round(base_buckets * lf));

    std::cout << "\n" << COLOR_YELLOW << "Load Factor: " << std::fixed
              << std::setprecision(2) << lf << " (" << num_keys << " keys)"
              << COLOR_RESET << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (uint32_t batch : batch_sizes) {
      config_num++;

      std::cout << "[" << config_num << "/" << total_configs << "] "
                << "Batch " << std::setw(6) << batch << " ... " << std::flush;

      SearchResult result = runSearchTest(num_keys, base_buckets, batch, lf);

      if (result.warp_time_ms < 0 || result.thread_time_ms < 0) {
        std::cout << COLOR_RED << "ERROR" << COLOR_RESET << std::endl;
        failed++;
        continue;
      }

      // Write to CSV
      appendCSVRow(csv, lf, num_keys, base_buckets, batch,
                   result.warp_time_ms, result.warp_throughput_mops,
                   result.thread_time_ms, result.thread_throughput_mops);

      // Print result
      std::string fastest;
      std::string fastest_color;
      double speedup;

      if (result.warp_time_ms < result.thread_time_ms) {
        fastest = "WARP";
        fastest_color = COLOR_BLUE;
        speedup = result.thread_time_ms / result.warp_time_ms;
      } else {
        fastest = "THREAD";
        fastest_color = COLOR_GREEN;
        speedup = result.warp_time_ms / result.thread_time_ms;
      }

      std::cout << "W: " << std::fixed << std::setprecision(4) << result.warp_time_ms << " ms"
                << " | T: " << result.thread_time_ms << " ms"
                << " | " << fastest_color << fastest << " " << std::setprecision(2) << speedup << "x"
                << COLOR_RESET << " ";

      if (result.correctness_passed) {
        std::cout << COLOR_GREEN << "✓" << COLOR_RESET << std::endl;
        passed++;
      } else {
        std::cout << COLOR_RED << "✗ FAILED" << COLOR_RESET << std::endl;
        failed++;
      }
    }
  }

  // Summary
  std::cout << "\n" << COLOR_BLUE << "=== Experiment Complete ===" << COLOR_RESET << std::endl;
  std::cout << "Passed: " << COLOR_GREEN << passed << COLOR_RESET << std::endl;
  std::cout << "Failed: " << (failed > 0 ? COLOR_RED : COLOR_GREEN)
            << failed << COLOR_RESET << std::endl;
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════════════════╗\n";
  std::cout << "║     GPU HashMap Search Performance Experiment              ║\n";
  std::cout << "╚════════════════════════════════════════════════════════════╝\n";

  // Print device info
  TestUtils::printDeviceInfo(0);

  // Open CSV file
  std::string csv_filename = "search_performance.csv";
  std::ofstream csv(csv_filename);

  if (!csv.is_open()) {
    std::cerr << COLOR_RED << "Error: Could not open " << csv_filename
              << " for writing" << COLOR_RESET << std::endl;
    return 1;
  }

  writeCSVHeader(csv);

  // Run experiment
  runExperiment(csv);

  csv.close();

  std::cout << "\n" << COLOR_GREEN << "Results saved to: " << csv_filename
            << COLOR_RESET << std::endl;
  std::cout << "\nCSV contains both warp-per-key and thread-per-key timings for direct comparison\n";
  std::cout << "\nSuggested analysis:\n";
  std::cout << "  1. Compare strategies: Plot warp vs thread throughput per configuration\n";
  std::cout << "  2. Load factor impact: How does LF affect each strategy?\n";
  std::cout << "  3. Batch size impact: When does thread-per-key become better?\n";
  std::cout << "  4. Speedup analysis: Which strategy wins at different LF/batch combinations?\n";
  std::cout << std::endl;

  return 0;
}
