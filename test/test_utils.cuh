/*
 * Test Utilities - Shared Helper Functions
 *
 * Common utilities for GPU hash map testing:
 *   - Key generation (sequential, random, colliding)
 *   - Device memory management helpers
 *   - Result verification
 *   - Test reporting
 */

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <chrono>
#include <unordered_set>

// ANSI color codes for test output
#define COLOR_GREEN "\033[1;32m"
#define COLOR_RED "\033[1;31m"
#define COLOR_YELLOW "\033[1;33m"
#define COLOR_BLUE "\033[1;34m"
#define COLOR_RESET "\033[0m"

namespace TestUtils {

// ============================================================================
// Key Generation Functions
// ============================================================================

/**
 * Generate sequential keys: 0, 1, 2, 3, ...
 */
template <typename KeyT>
void generateSequentialKeys(std::vector<KeyT>& keys, uint32_t count, KeyT start = 0) {
  keys.resize(count);
  for (uint32_t i = 0; i < count; i++) {
    keys[i] = start + i;
  }
}

/**
 * Generate random keys using uniform distribution
 */
template <typename KeyT>
void generateRandomKeys(std::vector<KeyT>& keys, uint32_t count, uint32_t seed = 42) {
  keys.resize(count);
  std::mt19937 gen(seed);
  std::uniform_int_distribution<KeyT> dis(0, std::numeric_limits<KeyT>::max());

  for (uint32_t i = 0; i < count; i++) {
    keys[i] = dis(gen);
  }
}

/**
 * Generate random unique keys (no duplicates)
 */
template <typename KeyT>
void generateRandomUniqueKeys(std::vector<KeyT>& keys, uint32_t count, uint32_t seed = 42) {
  std::unordered_set<KeyT> unique_keys;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<KeyT> dis(0, std::numeric_limits<KeyT>::max());

  while (unique_keys.size() < count) {
    unique_keys.insert(dis(gen));
  }

  keys.assign(unique_keys.begin(), unique_keys.end());
}

/**
 * Generate keys that hash to the same bucket (for collision testing)
 * Note: This requires knowledge of hash function, simplified version here
 */
template <typename KeyT>
void generateCollidingKeys(std::vector<KeyT>& keys, uint32_t count, uint32_t stride = 1000000) {
  keys.resize(count);
  // Generate keys with regular spacing to increase collision probability
  for (uint32_t i = 0; i < count; i++) {
    keys[i] = i * stride;
  }
}

/**
 * Generate values matching keys (value = key for easy verification)
 */
template <typename KeyT, typename ValueT>
void generateMatchingValues(std::vector<ValueT>& values, const std::vector<KeyT>& keys) {
  values.resize(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    values[i] = static_cast<ValueT>(keys[i]);
  }
}

/**
 * Generate constant values
 */
template <typename ValueT>
void generateConstantValues(std::vector<ValueT>& values, uint32_t count, ValueT value) {
  values.resize(count);
  std::fill(values.begin(), values.end(), value);
}

// ============================================================================
// Device Memory Management
// ============================================================================

/**
 * Allocate device memory and check for errors
 */
template <typename T>
T* allocateDeviceArray(uint32_t count) {
  T* d_ptr = nullptr;
  cudaError_t err = cudaMalloc(&d_ptr, count * sizeof(T));
  if (err != cudaSuccess) {
    std::cerr << COLOR_RED << "Failed to allocate device memory: "
              << cudaGetErrorString(err) << COLOR_RESET << std::endl;
    return nullptr;
  }
  return d_ptr;
}

/**
 * Free device memory
 */
template <typename T>
void freeDeviceArray(T* d_ptr) {
  if (d_ptr) {
    cudaFree(d_ptr);
  }
}

/**
 * Copy data from host to device
 */
template <typename T>
bool copyToDevice(T* d_dst, const std::vector<T>& h_src) {
  cudaError_t err = cudaMemcpy(d_dst, h_src.data(),
                               h_src.size() * sizeof(T),
                               cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << COLOR_RED << "Failed to copy to device: "
              << cudaGetErrorString(err) << COLOR_RESET << std::endl;
    return false;
  }
  return true;
}

/**
 * Copy data from device to host
 */
template <typename T>
bool copyToHost(std::vector<T>& h_dst, const T* d_src, uint32_t count) {
  h_dst.resize(count);
  cudaError_t err = cudaMemcpy(h_dst.data(), d_src,
                               count * sizeof(T),
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << COLOR_RED << "Failed to copy to host: "
              << cudaGetErrorString(err) << COLOR_RESET << std::endl;
    return false;
  }
  return true;
}

// ============================================================================
// Result Verification
// ============================================================================

/**
 * Verify search results match expected values
 * Returns number of mismatches
 */
template <typename ValueT>
uint32_t verifySearchResults(const std::vector<ValueT>& expected,
                              const std::vector<ValueT>& actual,
                              bool verbose = false) {
  uint32_t mismatches = 0;

  if (expected.size() != actual.size()) {
    std::cerr << COLOR_RED << "Size mismatch: expected " << expected.size()
              << " but got " << actual.size() << COLOR_RESET << std::endl;
    return expected.size();
  }

  for (size_t i = 0; i < expected.size(); i++) {
    if (expected[i] != actual[i]) {
      mismatches++;
      if (verbose && mismatches <= 10) {
        std::cerr << COLOR_YELLOW << "Mismatch at index " << i
                  << ": expected " << expected[i]
                  << " but got " << actual[i] << COLOR_RESET << std::endl;
      }
    }
  }

  if (mismatches > 0 && verbose) {
    if (mismatches > 10) {
      std::cerr << COLOR_YELLOW << "... and " << (mismatches - 10)
                << " more mismatches" << COLOR_RESET << std::endl;
    }
  }

  return mismatches;
}

/**
 * Count how many values match a specific value
 */
template <typename ValueT>
uint32_t countMatches(const std::vector<ValueT>& values, ValueT target) {
  uint32_t count = 0;
  for (const auto& val : values) {
    if (val == target) count++;
  }
  return count;
}

// ============================================================================
// Test Reporting
// ============================================================================

/**
 * Print test result with color coding
 */
void printTestResult(const std::string& test_name, bool passed,
                     const std::string& details = "") {
  if (passed) {
    std::cout << COLOR_GREEN << "✓ PASSED" << COLOR_RESET
              << " - " << test_name;
  } else {
    std::cout << COLOR_RED << "✗ FAILED" << COLOR_RESET
              << " - " << test_name;
  }

  if (!details.empty()) {
    std::cout << " (" << details << ")";
  }

  std::cout << std::endl;
}

/**
 * Print test section header
 */
void printSectionHeader(const std::string& section_name) {
  std::cout << "\n" << COLOR_BLUE << "=== " << section_name << " ==="
            << COLOR_RESET << std::endl;
}

/**
 * Print test summary
 */
void printTestSummary(uint32_t passed, uint32_t total) {
  std::cout << "\n" << COLOR_BLUE << "=== Test Summary ===" << COLOR_RESET << std::endl;

  double percentage = total > 0 ? (100.0 * passed / total) : 0.0;

  if (passed == total) {
    std::cout << COLOR_GREEN;
  } else if (passed > total / 2) {
    std::cout << COLOR_YELLOW;
  } else {
    std::cout << COLOR_RED;
  }

  std::cout << passed << "/" << total << " tests passed ("
            << std::fixed << std::setprecision(1) << percentage << "%)"
            << COLOR_RESET << std::endl;
}

// ============================================================================
// Performance Measurement
// ============================================================================

/**
 * Simple timer for performance measurement
 */
class Timer {
private:
  std::chrono::high_resolution_clock::time_point start_time_;

public:
  void start() {
    start_time_ = std::chrono::high_resolution_clock::now();
  }

  double elapsedMs() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time_);
    return duration.count() / 1000.0;
  }
};

/**
 * Calculate throughput in M ops/sec
 */
double calculateThroughput(uint32_t num_operations, double time_ms) {
  if (time_ms <= 0) return 0.0;
  return (num_operations / 1e6) / (time_ms / 1000.0);
}

/**
 * Print performance metrics
 */
void printPerformance(const std::string& operation,
                      uint32_t count,
                      double time_ms) {
  double throughput = calculateThroughput(count, time_ms);
  std::cout << "  " << operation << ": "
            << std::fixed << std::setprecision(3) << time_ms << " ms, "
            << std::fixed << std::setprecision(2) << throughput << " M ops/sec"
            << std::endl;
}

// ============================================================================
// Device Information
// ============================================================================

/**
 * Print CUDA device information
 */
void printDeviceInfo(int device_idx = 0) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_idx);

  std::cout << COLOR_BLUE << "Device: " << prop.name << COLOR_RESET << std::endl;
  std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "  Global Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
  std::cout << "  Warp Size: " << prop.warpSize << std::endl;
}

} // namespace TestUtils
