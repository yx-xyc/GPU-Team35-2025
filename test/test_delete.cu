/*
 * test_delete.cu
 * 
 * Tests for delete operation: 
 * - basic correctness 
 * - idempotence 
 * - non-existent keys 
 * - count consistency 
 * - search semantics
 */

#include "../include/gpu_hash_map.cuh"
#include "test_utils.cuh"
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>

template <typename KeyT=uint32_t, typename ValueT=uint32_t>
bool run_delete_suite(uint32_t num_buckets,
                      uint32_t num_keys,
                      uint32_t device_idx = 0,
                      uint32_t seed = 12345,
                      bool verbose = false) {
  std::cout << "[DeleteSuite] buckets=" << num_buckets
            << " keys=" << num_keys << std::endl;

  std::vector<KeyT>   h_keys(num_keys);
  std::vector<ValueT> h_values(num_keys);
  for (uint32_t i = 0; i < num_keys; ++i) {
    h_keys[i]   = static_cast<KeyT>(i);
    h_values[i] = static_cast<ValueT>(i * 10 + 1);
  }

  std::vector<KeyT> h_del;
  for (uint32_t i = 0; i < num_keys; ++i) if (i % 3 == 0) h_del.push_back(h_keys[i]);
  for (uint32_t i = 0; i < 16; ++i) h_del.push_back(static_cast<KeyT>(num_keys + 100 + i));

  std::vector<KeyT> h_kept;
  for (uint32_t i = 0; i < num_keys; ++i) if (i % 3 != 0) h_kept.push_back(h_keys[i]);

  KeyT*   d_keys     = allocateDeviceArray<KeyT>(num_keys);
  ValueT* d_values   = allocateDeviceArray<ValueT>(num_keys);
  KeyT*   d_del_keys = allocateDeviceArray<KeyT>(static_cast<uint32_t>(h_del.size()));
  ValueT* d_results  = allocateDeviceArray<ValueT>(num_keys);

  copyToDevice(d_keys,   h_keys.data(),   num_keys);
  copyToDevice(d_values, h_values.data(), num_keys);
  copyToDevice(d_del_keys, h_del.data(),  static_cast<uint32_t>(h_del.size()));

  GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, seed, verbose);
  map.buildTable(d_keys, d_values, num_keys);

  uint32_t cnt_before = map.countTable();
  if (cnt_before != num_keys) {
    std::cerr << "[DeleteSuite] Precondition failed: countTable=" << cnt_before
              << " expected=" << num_keys << std::endl;
    return false;
  }

  map.deleteTable(d_del_keys, static_cast<uint32_t>(h_del.size()));
  map.deleteTable(d_del_keys, static_cast<uint32_t>(h_del.size()));

  uint32_t cnt_after = map.countTable();
  uint32_t expected_after = static_cast<uint32_t>(h_kept.size());
  if (cnt_after != expected_after) {
    std::cerr << "[DeleteSuite] Count mismatch: got=" << cnt_after
              << " expected=" << expected_after << std::endl;
    return false;
  }

  map.searchTable(d_keys, d_results, num_keys);
  std::vector<ValueT> h_res;
  copyToHost(h_res, d_results, num_keys);

  bool ok = true;
  for (uint32_t i = 0; i < num_keys; ++i) {
    bool should_exist = (i % 3 != 0);
    ValueT expected = static_cast<ValueT>(i * 10 + 1);
    ValueT got = h_res[i];
    if (should_exist) {
      if (got != expected) {
        std::cerr << "[DeleteSuite] Search mismatch for kept key " << i
                  << ": got=" << got << " expected=" << expected << std::endl;
        ok = false;
        break;
      }
    } else {
      if (got != SEARCH_NOT_FOUND) {
        std::cerr << "[DeleteSuite] Deleted key " << i
                  << " should be SEARCH_NOT_FOUND, got=" << got << std::endl;
        ok = false;
        break;
      }
    }
  }

  GpuHashMap<KeyT, ValueT> empty_map(num_buckets, device_idx, seed, verbose);
  if (empty_map.countTable() != 0) ok = false;
  empty_map.deleteTable(d_del_keys, static_cast<uint32_t>(h_del.size()));
  if (empty_map.countTable() != 0) ok = false;

  freeDeviceArray(d_keys);
  freeDeviceArray(d_values);
  freeDeviceArray(d_del_keys);
  freeDeviceArray(d_results);
  return ok;
}

int main() {
  bool all_ok = true;
  all_ok = all_ok && run_delete_suite<uint32_t, uint32_t>(4096, 1000);
  all_ok = all_ok && run_delete_suite<uint32_t, uint32_t>(1<<15, 5000);
  if (all_ok) {
    std::cout << "✓ Delete tests passed!" << std::endl;
    return 0;
  } else {
    std::cout << "✗ Delete tests failed!" << std::endl;
    return 1;
  }
}

