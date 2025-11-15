/*
 * GPU Hash Map - Host-side implementation
 *
 * This class owns GPU memory and manages the hash table lifecycle.
 * It provides high-level operations that launch CUDA kernels.
 */

#pragma once

#include "hash_map_context.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <random>
#include <sstream>

/*
 * GpuHashMap - Host-side hash map class
 *
 * Template Parameters:
 *   KeyT - Key type
 *   ValueT - Value type
 *
 * Responsibilities:
 *   - Allocate and manage GPU memory
 *   - Launch kernels for bulk operations
 *   - Provide high-level API to users
 */
template <typename KeyT, typename ValueT>
class GpuHashMap {
 private:
  uint32_t num_buckets_;    // Number of buckets in hash table
  uint32_t device_idx_;     // GPU device index
  int64_t seed_;            // Random seed for hash function
  bool verbose_;            // Print debug info
  uint32_t search_warp_threshold_;  // Threshold for hybrid search (queries < threshold use one-warp-per-key)

  // Hash function parameters
  uint32_t hash_x_;
  uint32_t hash_y_;

  // Device pointers (owned by this class)
  KeyT* d_keys_;
  ValueT* d_values_;
  uint32_t* d_status_;

  // Device context (for kernel launches)
  GpuHashMapContext<KeyT, ValueT> context_;

  // Initialize hash function parameters
  void initHashFunction() {
    std::mt19937_64 rng(seed_);
    std::uniform_int_distribution<uint32_t> dist(1, GpuHashMapContext<KeyT, ValueT>::PRIME_DIVISOR - 1);
    hash_x_ = dist(rng);
    hash_y_ = dist(rng);
  }

 public:
  /*
   * Constructor
   *
   * Parameters:
   *   num_buckets - size of hash table
   *   device_idx - GPU device to use (default 0)
   *   seed - random seed for hash function (default 0)
   *   verbose - print debug information (default false)
   *   search_warp_threshold - queries below this use one-warp-per-key, above use one-thread-per-key (default 5000)
   */
  GpuHashMap(uint32_t num_buckets,
             uint32_t device_idx = 0,
             int64_t seed = 0,
             bool verbose = false,
             uint32_t search_warp_threshold = 5000)
      : num_buckets_(2 * num_buckets),
        device_idx_(device_idx),
        seed_(seed),
        verbose_(verbose),
        search_warp_threshold_(search_warp_threshold),
        d_keys_(nullptr),
        d_values_(nullptr),
        d_status_(nullptr) {

    // Set device
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

    // Initialize hash function
    initHashFunction();

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_keys_, sizeof(KeyT) * int(  num_buckets_) ));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values_, sizeof(ValueT) * int(  num_buckets_) ) );
    CHECK_CUDA_ERROR(cudaMalloc(&d_status_, sizeof(uint32_t) * int(  num_buckets_) ));

    // Initialize status array to EMPTY
    CHECK_CUDA_ERROR(cudaMemset(d_status_, 0, sizeof(uint32_t) * int(  num_buckets_) ));

    // Initialize context
    context_.initialize( num_buckets_, hash_x_, hash_y_, d_keys_, d_values_, d_status_);

    if (verbose_) {
      std::cout << toString() << std::endl;
    }
  }

  /*
   * Destructor - Free GPU memory
   */
  ~GpuHashMap() {
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    if (d_keys_) CHECK_CUDA_ERROR(cudaFree(d_keys_));
    if (d_values_) CHECK_CUDA_ERROR(cudaFree(d_values_));
    if (d_status_) CHECK_CUDA_ERROR(cudaFree(d_status_));
  }

  // Prevent copying (would duplicate ownership)
  GpuHashMap(const GpuHashMap&) = delete;
  GpuHashMap& operator=(const GpuHashMap&) = delete;

  /*
   * Get device context for custom kernels
   */
  GpuHashMapContext<KeyT, ValueT> getContext() const {
    return context_;
  }

  /*
   * Get hash table information as string
   */
  std::string toString() const {
    std::stringstream ss;
    ss << "=== GPU Hash Map ===" << std::endl;
    ss << "  Num buckets: " << num_buckets_ << std::endl;
    ss << "  Device: " << device_idx_ << std::endl;
    ss << "  Hash parameters: (" << hash_x_ << ", " << hash_y_ << ")" << std::endl;
    ss << "  Search warp threshold: " << search_warp_threshold_ << std::endl;
    ss << "  Memory: " << (sizeof(KeyT) + sizeof(ValueT) + sizeof(uint32_t)) * num_buckets_ / (1024.0 * 1024.0)
       << " MB" << std::endl;
    return ss.str();
  }

  // ============================================================================
  // High-level operations (kernel launchers)
  // These are declared here and implemented in src/kernels/*.cuh
  // ============================================================================

  /*
   * buildTable - Bulk insert operation
   *
   * Inserts multiple key-value pairs in parallel.
   *
   * Parameters:
   *   d_keys - device array of keys
   *   d_values - device array of values
   *   num_keys - number of keys to insert
   */
  void buildTable(const KeyT* d_keys, const ValueT* d_values, uint32_t num_keys);

  /*
   * searchTable - Bulk search operation
   *
   * Searches for multiple keys in parallel.
   * Uses hybrid strategy based on search_warp_threshold.
   *
   * Parameters:
   *   d_queries - device array of query keys
   *   d_results - device array for results (output)
   *   num_queries - number of queries
   */
  void searchTable(const KeyT* d_queries, ValueT* d_results, uint32_t num_queries);

  /*
   * searchTableWarpPerKey - Force warp-per-key search strategy
   *
   * Directly calls one-warp-per-key kernel, ignoring threshold.
   * Each warp searches ONE key cooperatively (32x parallelism per key).
   *
   * Parameters:
   *   d_queries - device array of query keys
   *   d_results - device array for results (output)
   *   num_queries - number of queries
   */
  void searchTableWarpPerKey(const KeyT* d_queries, ValueT* d_results, uint32_t num_queries);

  /*
   * searchTableThreadPerKey - Force thread-per-key search strategy
   *
   * Directly calls one-thread-per-key kernel, ignoring threshold.
   * Each thread searches its own key independently.
   *
   * Parameters:
   *   d_queries - device array of query keys
   *   d_results - device array for results (output)
   *   num_queries - number of queries
   */
  void searchTableThreadPerKey(const KeyT* d_queries, ValueT* d_results, uint32_t num_queries);

  /*
   * deleteTable - Bulk delete operation
   *
   * Deletes multiple keys in parallel.
   *
   * Parameters:
   *   d_keys - device array of keys to delete
   *   num_keys - number of keys to delete
   */
  void deleteTable(const KeyT* d_keys, uint32_t num_keys);

  /*
   * deleteTableOptimized - Bulk delete operation (optimized version)
   *
   * Uses optimized kernel with reduced atomic operations for better performance.
   *
   * Parameters:
   *   d_keys - device array of keys to delete
   *   num_keys - number of keys to delete
   */
  void deleteTableOptimized(const KeyT* d_keys, uint32_t num_keys);

  /*
   * countTable - Count total valid elements
   *
   * Returns the number of valid key-value pairs currently in the table.
   *
   * Returns:
   *   Number of occupied slots
   */
  uint32_t countTable();

  /*
   * countTableOptimized - Count total valid elements (optimized version)
   *
   * Uses block-level reduction to minimize atomic operations.
   *
   * Returns:
   *   Number of occupied slots
   */
  uint32_t countTableOptimized();

  /*
   * concurrentOperations - Mixed operations in one batch
   *
   * Performs insert, delete, and search operations concurrently.
   * Useful for benchmarking and testing.
   *
   * Parameters:
   *   d_insert_keys, d_insert_values - keys/values to insert
   *   num_inserts - number of insertions
   *   d_delete_keys - keys to delete
   *   num_deletes - number of deletions
   *   d_search_keys - keys to search
   *   d_search_results - output for search results
   *   num_searches - number of searches
   *
   * Note: Operations within a batch may have race conditions.
   *       Order of operations is not guaranteed.
   */
  void concurrentOperations(const KeyT* d_insert_keys,
                           const ValueT* d_insert_values,
                           uint32_t num_inserts,
                           const KeyT* d_delete_keys,
                           uint32_t num_deletes,
                           const KeyT* d_search_keys,
                           ValueT* d_search_results,
                           uint32_t num_searches);

  /*
   * clear - Remove all entries
   *
   * Resets all slots to EMPTY status.
   */
  void clear() {
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    CHECK_CUDA_ERROR(cudaMemset(d_status_, 0, sizeof(uint32_t) * num_buckets_));
  }

  /*
   * getIterator - Create iterator for traversing all key-value pairs
   *
   * Returns an iterator that allows sequential access to all valid entries.
   * The iterator copies data from device to host, so it's best used for
   * small tables or when you need to inspect all entries.
   *
   * Returns:
   *   GpuHashMapIterator instance
   *
   * Usage:
   *   auto iter = hash_map.getIterator();
   *   while (iter.hasNext()) {
   *     auto pair = iter.next();
   *     std::cout << pair.key << " -> " << pair.value << std::endl;
   *   }
   */
  GpuHashMapIterator<KeyT, ValueT> getIterator() {
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    return GpuHashMapIterator<KeyT, ValueT>(d_keys_, d_values_, d_status_, num_buckets_);
  }

  /*
   * getNumBuckets - Get table size
   */
  uint32_t getNumBuckets() const {
    return num_buckets_;
  }

  /*
   * getDeviceIdx - Get GPU device index
   */
  uint32_t getDeviceIdx() const {
    return device_idx_;
  }
};
