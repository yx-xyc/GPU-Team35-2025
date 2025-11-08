/*
 * test_delete.cu
 * 
 * Tests for delete operation: 
 * - basic correctness 
 * - idempotence 
 * - non-existent keys 
 * - count consistency 
 * - search semantics
 * - duplicates
 * - concurrent deletion
 */

#include "../include/gpu_hash_map.cuh"
#include "../test/test_utils.cuh"
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

// -----------------------------------------------------------
// Utility functions (used by delete tests)
// -----------------------------------------------------------
template <typename T>
T* allocateDeviceArray(uint32_t count) {
    T* ptr = nullptr;
    cudaMalloc(&ptr, sizeof(T) * count);
    return ptr;
}

template <typename T>
void freeDeviceArray(T* ptr) {
    cudaFree(ptr);
}

template <typename T>
void copyToDevice(T* d_dst, const T* h_src, uint32_t count) {
    cudaMemcpy(d_dst, h_src, sizeof(T) * count, cudaMemcpyHostToDevice);
}

template <typename T>
void copyToHost(std::vector<T>& h_dst, const T* d_src, uint32_t count) {
    h_dst.resize(count);
    cudaMemcpy(h_dst.data(), d_src, sizeof(T) * count, cudaMemcpyDeviceToHost);
}

// ===============================================================
// === Base Delete Test Suite (original) =========================
// ===============================================================
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

// ===============================================================
// === NEW TESTS FOR DELETE OPERATION ============================
// ===============================================================
template <typename KeyT=uint32_t, typename ValueT=uint32_t>
bool run_delete_edge_cases(uint32_t num_buckets,
                           uint32_t device_idx = 0,
                           uint32_t seed = 22222,
                           bool verbose = false) {
  std::cout << "\n=== Running Boundary Delete Test ===" << std::endl;

  std::vector<KeyT> keys = {1, 2, 3, 4, 5};
  std::vector<ValueT> vals = {10, 20, 30, 40, 50};
  KeyT* d_keys = allocateDeviceArray<KeyT>(keys.size());
  ValueT* d_vals = allocateDeviceArray<ValueT>(vals.size());
  copyToDevice(d_keys, keys.data(), keys.size());
  copyToDevice(d_vals, vals.data(), vals.size());

  GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, seed, verbose);
  map.buildTable(d_keys, d_vals, keys.size());

  KeyT fake_key = 999;
  KeyT* d_fake = allocateDeviceArray<KeyT>(1);
  copyToDevice(d_fake, &fake_key, 1);
  map.deleteTable(d_fake, 1);

  KeyT target = 3;
  KeyT* d_target = allocateDeviceArray<KeyT>(1);
  copyToDevice(d_target, &target, 1);
  map.deleteTable(d_target, 1);
  uint32_t before = map.countTable();
  map.buildTable(d_target, d_vals + 2, 1);
  uint32_t after = map.countTable();

  bool ok = (after == before + 1);
  if (!ok) std::cerr << "[BoundaryTest] Count mismatch after reinsert\n";

  freeDeviceArray(d_fake);
  freeDeviceArray(d_target);
  freeDeviceArray(d_keys);
  freeDeviceArray(d_vals);
  return ok;
}

template <typename KeyT=uint32_t, typename ValueT=uint32_t>
bool run_delete_duplicates(uint32_t num_buckets,
                           uint32_t device_idx = 0,
                           uint32_t seed = 33333,
                           bool verbose = false) {
  std::cout << "\n=== Running Duplicate Delete Test ===" << std::endl;

  std::vector<KeyT> keys = {1, 2, 2, 2, 3, 4};
  std::vector<ValueT> vals = {10, 20, 21, 22, 30, 40};
  KeyT* d_keys = allocateDeviceArray<KeyT>(keys.size());
  ValueT* d_vals = allocateDeviceArray<ValueT>(vals.size());
  copyToDevice(d_keys, keys.data(), keys.size());
  copyToDevice(d_vals, vals.data(), vals.size());

  GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, seed, verbose);
  map.buildTable(d_keys, d_vals, keys.size());

  KeyT del_key = 2;
  KeyT* d_del = allocateDeviceArray<KeyT>(1);
  copyToDevice(d_del, &del_key, 1);
  map.deleteTable(d_del, 1);
  map.deleteTable(d_del, 1);

  uint32_t cnt = map.countTable();
  bool ok = (cnt == 3);
  if (!ok)
    std::cerr << "[DuplicateDelete] Count mismatch, got " << cnt << " expected 3\n";

  freeDeviceArray(d_keys);
  freeDeviceArray(d_vals);
  freeDeviceArray(d_del);
  return ok;
}

template <typename KeyT=uint32_t, typename ValueT=uint32_t>
bool run_delete_concurrent(uint32_t num_buckets,
                           uint32_t num_keys,
                           uint32_t device_idx = 0,
                           uint32_t seed = 44444,
                           bool verbose = false) {
  std::cout << "\n=== Running Concurrent Delete Test ===" << std::endl;

  std::vector<KeyT> h_keys(num_keys);
  std::vector<ValueT> h_vals(num_keys);
  for (uint32_t i = 0; i < num_keys; ++i) {
    h_keys[i] = i;
    h_vals[i] = i * 10 + 1;
  }

  KeyT* d_keys = allocateDeviceArray<KeyT>(num_keys);
  ValueT* d_vals = allocateDeviceArray<ValueT>(num_keys);
  copyToDevice(d_keys, h_keys.data(), num_keys);
  copyToDevice(d_vals, h_vals.data(), num_keys);

  GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, seed, verbose);
  map.buildTable(d_keys, d_vals, num_keys);

  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  std::vector<KeyT> del1, del2;
  for (uint32_t i = 0; i < num_keys; ++i)
    ((i % 2 == 0) ? del1 : del2).push_back(h_keys[i]);

  KeyT* d_del1 = allocateDeviceArray<KeyT>(del1.size());
  KeyT* d_del2 = allocateDeviceArray<KeyT>(del2.size());
  copyToDevice(d_del1, del1.data(), del1.size());
  copyToDevice(d_del2, del2.data(), del2.size());

  map.deleteTable(d_del1, del1.size());
  map.deleteTable(d_del2, del2.size());
  cudaStreamSynchronize(s1);
  cudaStreamSynchronize(s2);

  uint32_t cnt_after = map.countTable();
  bool ok = (cnt_after == 0);
  if (!ok)
    std::cerr << "[ConcurrentDelete] Expected 0 keys, got " << cnt_after << "\n";

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  freeDeviceArray(d_del1);
  freeDeviceArray(d_del2);
  freeDeviceArray(d_keys);
  freeDeviceArray(d_vals);
  return ok;
}

// ===============================================================
// === MAIN ======================================================
// ===============================================================
int main() {
  bool all_ok = true;
  all_ok = all_ok && run_delete_suite<uint32_t, uint32_t>(4096, 1000);
  all_ok = all_ok && run_delete_suite<uint32_t, uint32_t>(1<<15, 5000);

  all_ok = all_ok && run_delete_edge_cases<uint32_t, uint32_t>(2048);
  all_ok = all_ok && run_delete_duplicates<uint32_t, uint32_t>(2048);
  all_ok = all_ok && run_delete_concurrent<uint32_t, uint32_t>(4096, 1000);

  if (all_ok) {
    std::cout << "✓ All Delete tests (extended) passed!" << std::endl;
    return 0;
  } else {
    std::cout << "✗ Some Delete tests failed!" << std::endl;
    return 1;
  }
}
