/*
 * GPU Hash Map Context - Device-side class
 *
 * This class does NOT own memory and is designed to be shallow-copied to device.
 * It provides warp-cooperative operations for use in CUDA kernels.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Slot status flags
enum SlotStatus : uint32_t {
  EMPTY = 0,      // Slot is empty
  OCCUPIED = 1,   // Slot contains valid key-value pair
  TOMBSTONE = 2,  // Slot was deleted (for open addressing)
  PENDING = 3     // Slot is being written (key not ready yet)
};

/*
 * GpuHashMapContext - Device-side context class
 *
 * Template Parameters:
 *   KeyT - Key type (must be compatible with bitwise operations)
 *   ValueT - Value type
 *
 * Design Pattern (from SlabHash):
 *   - Host class (GpuHashMap) owns GPU memory
 *   - Context class (this) does not own memory, used in kernels
 *   - Shallow copy to device for kernel use
 */
template <typename KeyT, typename ValueT>
class GpuHashMapContext {
 public:
  // Constants
  static constexpr uint32_t WARP_WIDTH = 32;
  static constexpr uint32_t PRIME_DIVISOR = 4294967291u;  // Large prime for hashing

  // Maximum linear probing distance before giving up
  static constexpr uint32_t MAX_PROBE_LENGTH = 128;

 private:
  uint32_t num_buckets_;   // Number of buckets in hash table
  uint32_t hash_x_;        // Hash function parameter
  uint32_t hash_y_;        // Hash function parameter

  // Device pointers (not owned by this class)
  KeyT* d_keys_;           // Array of keys
  ValueT* d_values_;       // Array of values
  uint32_t* d_status_;     // Array of slot status flags

 public:
  // Constructors
  __host__ __device__ GpuHashMapContext()
      : num_buckets_(0),
        hash_x_(0),
        hash_y_(0),
        d_keys_(nullptr),
        d_values_(nullptr),
        d_status_(nullptr) {}

  __host__ __device__ GpuHashMapContext(const GpuHashMapContext& other)
      : num_buckets_(other.num_buckets_),
        hash_x_(other.hash_x_),
        hash_y_(other.hash_y_),
        d_keys_(other.d_keys_),
        d_values_(other.d_values_),
        d_status_(other.d_status_) {}

  __host__ __device__ ~GpuHashMapContext() {}

  // Initialize context parameters (called from host)
  __host__ void initialize(uint32_t num_buckets,
                          uint32_t hash_x,
                          uint32_t hash_y,
                          KeyT* d_keys,
                          ValueT* d_values,
                          uint32_t* d_status) {
    num_buckets_ = num_buckets;
    hash_x_ = hash_x;
    hash_y_ = hash_y;
    d_keys_ = d_keys;
    d_values_ = d_values;
    d_status_ = d_status;
  }

  // Accessors
  __device__ __host__ __forceinline__ uint32_t getNumBuckets() const {
    return num_buckets_;
  }

  __device__ __host__ __forceinline__ KeyT* getKeys() const {
    return d_keys_;
  }

  __device__ __host__ __forceinline__ ValueT* getValues() const {
    return d_values_;
  }

  __device__ __host__ __forceinline__ uint32_t* getStatus() const {
    return d_status_;
  }

  // Hash function - Universal hashing
  // Returns bucket index for given key
  __device__ __host__ __forceinline__ uint32_t computeBucket(const KeyT& key) const {
    // Universal hash: ((a * key + b) mod p) mod m
    // where p is prime, m is table size
    uint64_t hash = ((uint64_t)hash_x_ ^ (uint64_t)key) + (uint64_t)hash_y_;
    return (hash % PRIME_DIVISOR) % num_buckets_;
  }

  // ============================================================================
  // Warp-cooperative operations
  // These are declared here and implemented in src/warp/*.cuh
  // ============================================================================

  /*
   * insertKey - Warp-cooperative insert operation
   *
   * All 32 threads in a warp cooperate to insert key-value pairs.
   * Uses linear probing for collision resolution.
   *
   * Parameters:
   *   to_insert - true if this thread has a key to insert
   *   laneId - lane ID within warp (0-31)
   *   key - key to insert
   *   value - value to insert
   *   bucket - starting bucket (from computeBucket)
   */
  __device__ __forceinline__ void insertKey(bool to_insert,
                                            const uint32_t laneId,
                                            const KeyT& key,
                                            const ValueT& value,
                                            uint32_t bucket);

  /*
   * searchKey - Warp-cooperative search operation
   *
   * All 32 threads in a warp cooperate to search for keys.
   *
   * Parameters:
   *   to_search - true if this thread has a key to search
   *   laneId - lane ID within warp (0-31)
   *   key - key to search for
   *   result - output: value if found, SEARCH_NOT_FOUND otherwise
   *   bucket - starting bucket (from computeBucket)
   */
  __device__ __forceinline__ void searchKey(bool to_search,
                                            const uint32_t laneId,
                                            const KeyT& key,
                                            ValueT& result,
                                            uint32_t bucket);

  /*
   * deleteKey - Warp-cooperative delete operation
   *
   * All 32 threads in a warp cooperate to delete keys.
   * Uses tombstone marking (sets status to TOMBSTONE).
   *
   * Parameters:
   *   to_delete - true if this thread has a key to delete
   *   laneId - lane ID within warp (0-31)
   *   key - key to delete
   *   bucket - starting bucket (from computeBucket)
   */
  __device__ __forceinline__ void deleteKey(bool to_delete,
                                            const uint32_t laneId,
                                            const KeyT& key,
                                            uint32_t bucket);

  /*
   * countKey - Warp-cooperative count operation
   *
   * Counts how many times a key appears (should be 0 or 1).
   *
   * Parameters:
   *   to_count - true if this thread has a key to count
   *   laneId - lane ID within warp (0-31)
   *   key - key to count
   *   count - output: number of occurrences
   *   bucket - starting bucket (from computeBucket)
   */
  __device__ __forceinline__ void countKey(bool to_count,
                                           const uint32_t laneId,
                                           const KeyT& key,
                                           uint32_t& count,
                                           uint32_t bucket);
};
