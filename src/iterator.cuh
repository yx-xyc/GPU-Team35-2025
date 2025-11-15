/*
 * Hash Map Iterator
 *
 * Provides ability to iterate through all valid key-value pairs.
 *
 * DESIGN:
 *   - Simple sequential iterator (not parallel)
 *   - Skips EMPTY and TOMBSTONE slots
 *   - Returns only OCCUPIED slots
 *
 * USAGE:
 *   GpuHashMapIterator<uint32_t, uint32_t> iter = hash_map.getIterator();
 *   while (iter.hasNext()) {
 *     auto pair = iter.next();
 *     std::cout << pair.key << " -> " << pair.value << std::endl;
 *   }
 *
 * IMPLEMENTATION NOTE:
 *   Current design copies all data to host (simple but slow).
 *   Better: Parallel GPU iterator or stream data in chunks.
 *
 * REFERENCES:
 *   - SlabHash/src/slab_iterator.cuh (for slab-based version)
 */

#pragma once

#include "hash_map_context.cuh"
#include <vector>
#include <utility>

/*
 * Key-value pair structure
 */
template <typename KeyT, typename ValueT>
struct KeyValuePair {
  KeyT key;
  ValueT value;

  KeyValuePair(KeyT k, ValueT v) : key(k), value(v) {}
};

/*
 * Hash Map Iterator
 *
 * Template Parameters:
 *   KeyT - Key type
 *   ValueT - Value type
 */
template <typename KeyT, typename ValueT>
class GpuHashMapIterator {
 private:
  std::vector<KeyValuePair<KeyT, ValueT>> pairs_;
  size_t current_index_;

 public:
  /*
   * Constructor: Fetch all valid pairs from GPU
   *
   * Parameters:
   *   d_keys, d_values, d_status - device arrays
   *   num_buckets - size of hash table
   */
  GpuHashMapIterator(const KeyT* d_keys,
                     const ValueT* d_values,
                     const uint32_t* d_status,
                     uint32_t num_buckets)
      : current_index_(0) {

    // Allocate host arrays for temporary storage
    KeyT* h_keys = new KeyT[num_buckets];
    ValueT* h_values = new ValueT[num_buckets];
    uint32_t* h_status = new uint32_t[num_buckets];

    // Copy data from device to host
    cudaError_t err;
    err = cudaMemcpy(h_keys, d_keys, num_buckets * sizeof(KeyT), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      delete[] h_keys;
      delete[] h_values;
      delete[] h_status;
      throw std::runtime_error("Failed to copy keys from device: " +
                               std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(h_values, d_values, num_buckets * sizeof(ValueT), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      delete[] h_keys;
      delete[] h_values;
      delete[] h_status;
      throw std::runtime_error("Failed to copy values from device: " +
                               std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(h_status, d_status, num_buckets * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      delete[] h_keys;
      delete[] h_values;
      delete[] h_status;
      throw std::runtime_error("Failed to copy status from device: " +
                               std::string(cudaGetErrorString(err)));
    }

    // Collect all OCCUPIED entries into pairs_ vector
    for (uint32_t i = 0; i < num_buckets; i++) {
      if (h_status[i] == OCCUPIED) {
        pairs_.push_back(KeyValuePair<KeyT, ValueT>(h_keys[i], h_values[i]));
      }
    }

    // Free temporary host arrays
    delete[] h_keys;
    delete[] h_values;
    delete[] h_status;
  }

  /*
   * Check if more pairs available
   */
  bool hasNext() const {
    return current_index_ < pairs_.size();
  }

  /*
   * Get next key-value pair
   *
   * Returns:
   *   KeyValuePair containing key and value
   */
  KeyValuePair<KeyT, ValueT> next() {
    if (!hasNext()) {
      throw std::runtime_error("No more elements in iterator");
    }
    return pairs_[current_index_++];
  }

  /*
   * Get total number of valid pairs
   */
  size_t size() const {
    return pairs_.size();
  }

  /*
   * Reset iterator to beginning
   */
  void reset() {
    current_index_ = 0;
  }
};
