/*
 * Simple Debug Test for Count Function
 */

#include "gpu_hash_map.cuh"
#include <iostream>
#include <vector>

using KeyT = uint32_t;
using ValueT = uint32_t;

int main() {
  std::cout << "=== Count Function Debug Test ===" << std::endl << std::endl;

  // Configuration
  const uint32_t num_buckets = 100;  // Small table for debugging
  const uint32_t num_keys = 10;      // Only 10 keys
  
  std::cout << "Creating hash map with " << num_buckets << " buckets..." << std::endl;
  GpuHashMap<KeyT, ValueT> map(num_buckets, 0, 12345, false);

  // Create simple keys: 1, 2, 3, ..., 10
  std::vector<KeyT> h_keys = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<ValueT> h_values = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};

  // Copy to device
  KeyT* d_keys;
  ValueT* d_values;
  CHECK_CUDA_ERROR(cudaMalloc(&d_keys, num_keys * sizeof(KeyT)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_values, num_keys * sizeof(ValueT)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(KeyT),
                               cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values.data(), num_keys * sizeof(ValueT),
                               cudaMemcpyHostToDevice));

  // Insert keys
  std::cout << "\nStep 1: Inserting " << num_keys << " keys..." << std::endl;
  map.buildTable(d_keys, d_values, num_keys);
  std::cout << "  Insert complete" << std::endl;

  // Manually check what's in the table
  std::cout << "\nStep 2: Manual inspection of table..." << std::endl;
  auto ctx = map.getContext();
  
  KeyT* h_all_keys = new KeyT[num_buckets];
  ValueT* h_all_values = new ValueT[num_buckets];
  uint32_t* h_all_status = new uint32_t[num_buckets];
  
  CHECK_CUDA_ERROR(cudaMemcpy(h_all_keys, ctx.getKeys(), 
                               num_buckets * sizeof(KeyT),
                               cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(h_all_values, ctx.getValues(), 
                               num_buckets * sizeof(ValueT),
                               cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(h_all_status, ctx.getStatus(), 
                               num_buckets * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));
  
  std::cout << "  Slot contents (status enum: 0=EMPTY, 1=OCCUPIED, 2=TOMBSTONE):" << std::endl;
  uint32_t manual_count = 0;
  for (uint32_t i = 0; i < num_buckets; i++) {
    if (h_all_status[i] != 0) {  // Not EMPTY
      std::cout << "    Slot[" << i << "]: status=" << h_all_status[i] 
                << ", key=" << h_all_keys[i]
                << ", value=" << h_all_values[i] << std::endl;
      
      if (h_all_status[i] == 1) {  // OCCUPIED
        manual_count++;
      }
    }
  }
  std::cout << "  Manual count of OCCUPIED slots: " << manual_count << std::endl;

  // Test countTable()
  std::cout << "\nStep 3: Testing countTable() function..." << std::endl;
  uint32_t count = map.countTable();
  std::cout << "  countTable() returned: " << count << std::endl;

  // Test search
  std::cout << "\nStep 4: Testing search for inserted keys..." << std::endl;
  ValueT* d_results;
  CHECK_CUDA_ERROR(cudaMalloc(&d_results, num_keys * sizeof(ValueT)));
  map.searchTable(d_keys, d_results, num_keys);
  
  std::vector<ValueT> h_results(num_keys);
  CHECK_CUDA_ERROR(cudaMemcpy(h_results.data(), d_results, num_keys * sizeof(ValueT),
                               cudaMemcpyDeviceToHost));
  
  uint32_t found_count = 0;
  for (uint32_t i = 0; i < num_keys; i++) {
    if (h_results[i] == h_values[i]) {
      found_count++;
    } else {
      std::cout << "  ✗ Key " << h_keys[i] << ": expected value " << h_values[i] 
                << ", got " << h_results[i] << std::endl;
    }
  }
  std::cout << "  Search found: " << found_count << " / " << num_keys << std::endl;

  // Results
  std::cout << "\n=== RESULTS ===" << std::endl;
  std::cout << "Expected count: " << num_keys << std::endl;
  std::cout << "Manual count:   " << manual_count << std::endl;
  std::cout << "countTable():   " << count << std::endl;
  std::cout << "Search found:   " << found_count << std::endl;

  bool pass = (manual_count == num_keys && count == num_keys && found_count == num_keys);
  
  if (pass) {
    std::cout << "\n✓ ALL TESTS PASSED!" << std::endl;
  } else {
    std::cout << "\n✗ TESTS FAILED!" << std::endl;
    std::cout << "This means:" << std::endl;
    if (manual_count != num_keys) {
      std::cout << "  - Insert is not working correctly (keys not being written)" << std::endl;
    }
    if (count != manual_count) {
      std::cout << "  - countTable() is not counting correctly" << std::endl;
    }
    if (found_count != manual_count) {
      std::cout << "  - Search is not working correctly" << std::endl;
    }
  }

  // Cleanup
  delete[] h_all_keys;
  delete[] h_all_values;
  delete[] h_all_status;
  CHECK_CUDA_ERROR(cudaFree(d_keys));
  CHECK_CUDA_ERROR(cudaFree(d_values));
  CHECK_CUDA_ERROR(cudaFree(d_results));

  return pass ? 0 : 1;
}