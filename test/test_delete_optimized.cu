/*
 * Test for Optimized Delete Functionality
 *
 * This test verifies that the optimized delete implementation works correctly
 * and provides performance comparison with the standard delete operation.
 */

#include "gpu_hash_map.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>

using namespace std;
using namespace std::chrono;

// Test parameters
const uint32_t NUM_KEYS = 100000;
const uint32_t TABLE_SIZE = 200000;

/*
 * Test correctness of optimized delete operation
 */
void test_delete_correctness() {
    cout << "=== Testing Delete Correctness ===" << endl;
    
    GpuHashMap<uint32_t, uint32_t> hash_map(TABLE_SIZE, 0, 42, true);
    
    // Generate test data
    vector<uint32_t> keys(NUM_KEYS);
    vector<uint32_t> values(NUM_KEYS);
    
    for (uint32_t i = 0; i < NUM_KEYS; i++) {
        keys[i] = i + 1; // Avoid 0 as key
        values[i] = i * 2;
    }
    
    // Allocate device memory
    uint32_t* d_keys;
    uint32_t* d_values;
    uint32_t* d_results;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_keys, NUM_KEYS * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values, NUM_KEYS * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, NUM_KEYS * sizeof(uint32_t)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, keys.data(), NUM_KEYS * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, values.data(), NUM_KEYS * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    // Insert all keys
    hash_map.buildTable(d_keys, d_values, NUM_KEYS);
    
    // Verify count before delete
    uint32_t count_before = hash_map.countTableOptimized();
    cout << "Count before delete: " << count_before << endl;
    assert(count_before == NUM_KEYS);
    
    // Delete half of the keys using optimized method
    uint32_t keys_to_delete = NUM_KEYS / 2;
    hash_map.deleteTableOptimized(d_keys, keys_to_delete);
    
    // Verify count after delete
    uint32_t count_after = hash_map.countTableOptimized();
    cout << "Count after delete: " << count_after << endl;
    assert(count_after == NUM_KEYS - keys_to_delete);
    
    // Search for deleted keys (should not be found)
    hash_map.searchTable(d_keys, d_results, keys_to_delete);
    
    vector<uint32_t> search_results(keys_to_delete);
    CHECK_CUDA_ERROR(cudaMemcpy(search_results.data(), d_results, keys_to_delete * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    for (uint32_t i = 0; i < keys_to_delete; i++) {
        assert(search_results[i] == SEARCH_NOT_FOUND);
    }
    
    // Search for remaining keys (should be found)
    hash_map.searchTable(d_keys + keys_to_delete, d_results, NUM_KEYS - keys_to_delete);
    
    vector<uint32_t> remaining_results(NUM_KEYS - keys_to_delete);
    CHECK_CUDA_ERROR(cudaMemcpy(remaining_results.data(), d_results, (NUM_KEYS - keys_to_delete) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    for (uint32_t i = 0; i < NUM_KEYS - keys_to_delete; i++) {
        assert(remaining_results[i] == values[i + keys_to_delete]);
    }
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_keys));
    CHECK_CUDA_ERROR(cudaFree(d_values));
    CHECK_CUDA_ERROR(cudaFree(d_results));
    
    cout << "✓ Delete correctness test passed!" << endl;
}

/*
 * Performance comparison between standard and optimized delete
 */
void test_delete_performance() {
    cout << "\n=== Testing Delete Performance ===" << endl;
    
    const int NUM_ITERATIONS = 5;
    
    // Test data
    vector<uint32_t> keys(NUM_KEYS);
    vector<uint32_t> values(NUM_KEYS);
    
    for (uint32_t i = 0; i < NUM_KEYS; i++) {
        keys[i] = i + 1;
        values[i] = i * 2;
    }
    
    uint32_t* d_keys;
    uint32_t* d_values;
    CHECK_CUDA_ERROR(cudaMalloc(&d_keys, NUM_KEYS * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values, NUM_KEYS * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, keys.data(), NUM_KEYS * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, values.data(), NUM_KEYS * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    double standard_time = 0.0;
    double optimized_time = 0.0;
    
    // Test standard delete
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        GpuHashMap<uint32_t, uint32_t> hash_map(TABLE_SIZE, 0, 42 + iter);
        hash_map.buildTable(d_keys, d_values, NUM_KEYS);
        
        auto start = high_resolution_clock::now();
        hash_map.deleteTable(d_keys, NUM_KEYS / 2);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        auto end = high_resolution_clock::now();
        
        standard_time += duration_cast<microseconds>(end - start).count();
    }
    
    // Test optimized delete
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        GpuHashMap<uint32_t, uint32_t> hash_map(TABLE_SIZE, 0, 42 + iter);
        hash_map.buildTable(d_keys, d_values, NUM_KEYS);
        
        auto start = high_resolution_clock::now();
        hash_map.deleteTableOptimized(d_keys, NUM_KEYS / 2);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        auto end = high_resolution_clock::now();
        
        optimized_time += duration_cast<microseconds>(end - start).count();
    }
    
    standard_time /= NUM_ITERATIONS;
    optimized_time /= NUM_ITERATIONS;
    
    cout << "Standard delete time: " << standard_time << " μs" << endl;
    cout << "Optimized delete time: " << optimized_time << " μs" << endl;
    cout << "Speedup: " << (standard_time / optimized_time) << "x" << endl;
    
    CHECK_CUDA_ERROR(cudaFree(d_keys));
    CHECK_CUDA_ERROR(cudaFree(d_values));
}

int main() {
    cout << "GPU Hash Map - Optimized Delete Test" << endl;
    cout << "====================================" << endl;
    
    // Set device
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    
    // Run tests
    test_delete_correctness();
    test_delete_performance();
    
    cout << "\n✓ All tests passed!" << endl;
    return 0;
}
