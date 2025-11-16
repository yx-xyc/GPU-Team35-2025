/*
 * Insert Performance Comparison Test (GPU vs. CPU)
 */

#include "gpu_hash_map.cuh" // Your hash map
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>       // for std::iota
#include <random>        // for std::mt19937
#include <unordered_map> // For CPU comparison
#include <unordered_set> // For unique key counting

using KeyT = uint32_t;
using ValueT = uint32_t;

// Timer class from your example
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start_;
        return elapsed.count();
    }
};

/**
 * @brief Runs a performance comparison for the insert operation.
 * @param num_keys Number of keys to insert.
 * @param num_buckets Number of buckets in the hash table.
 * @param use_random_keys If true, use random keys (harder); if false, use sequential keys (easier).
 */
void test_insert_performance(uint32_t num_keys, uint32_t num_buckets, bool use_random_keys) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "Testing Insert: " << num_keys << " keys, "
              << num_buckets << " buckets" << std::endl;
    std::cout << "Key Type:       " << (use_random_keys ? "Random (w/ duplicates)" : "Sequential (no duplicates)") << std::endl;
    std::cout << "Load factor:    " << std::fixed << std::setprecision(2)
              << (float)num_keys / num_buckets << std::endl;
    std::cout << "============================================================" << std::endl;

    const int num_runs = 10; // Average over 10 runs
    Timer timer;
    uint32_t seed = 12345;
    uint32_t device_idx = 0;

    // --- 1. Generate Host Data ---
    // We generate data *once* so both CPU and GPU test the exact same inputs
    std::vector<KeyT> h_keys(num_keys);
    std::vector<ValueT> h_values(num_keys);
    uint32_t unique_key_count = 0;

    if (use_random_keys) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> dist(1, num_keys / 2); // Force collisions
        std::unordered_set<KeyT> unique_set;
        for (uint32_t i = 0; i < num_keys; i++) {
            h_keys[i] = dist(rng);
            h_values[i] = i;
            unique_set.insert(h_keys[i]);
        }
        unique_key_count = unique_set.size();
    } else {
        std::iota(h_keys.begin(), h_keys.end(), 1); // 1, 2, 3, ...
        std::iota(h_values.begin(), h_values.end(), 1000); // 1000, 1001, ...
        unique_key_count = num_keys;
    }

    // --- 2. Test CPU (std::unordered_map) ---
    std::cout << "\n--- CPU (std::unordered_map) ---" << std::endl;
    double total_cpu_time = 0;
    uint32_t result_cpu = 0;

    for (int i = 0; i < num_runs; i++) {
        // Must create a new map each time, as this is part of "build"
        std::unordered_map<KeyT, ValueT> cpu_map;
        cpu_map.reserve(num_buckets); // Be fair: give it the same bucket hint

        timer.start();
        for (uint32_t j = 0; j < num_keys; j++) {
            cpu_map[h_keys[j]] = h_values[j]; // Insert or overwrite
        }
        total_cpu_time += timer.elapsed_ms();
        
        if (i == 0) {
            result_cpu = cpu_map.size(); // Only need count from one run
        }
    }
    double avg_cpu = total_cpu_time / num_runs;
    double throughput_cpu = (num_keys / (avg_cpu / 1000.0)) / 1e6; // Million Keys/sec

    std::cout << "Count: " << result_cpu << std::endl;
    std::cout << "Time:  " << std::fixed << std::setprecision(4)
              << avg_cpu << " ms" << std::endl;
    std::cout << "Perf:  " << std::fixed << std::setprecision(2)
              << throughput_cpu << " MK/s" << std::endl;

    // --- 3. Test GPU (GpuHashMap) ---
    std::cout << "\n--- GPU (GpuHashMap) ---" << std::endl;
    
    // Copy data to device *once*
    KeyT* d_keys;
    ValueT* d_values;
    CHECK_CUDA_ERROR(cudaMalloc(&d_keys, num_keys * sizeof(KeyT)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values, num_keys * sizeof(ValueT)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(),
                                num_keys * sizeof(KeyT),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values.data(),
                                num_keys * sizeof(ValueT),
                                cudaMemcpyHostToDevice));

    double total_gpu_time = 0;
    uint32_t result_gpu = 0;

    for (int i = 0; i < num_runs; i++) {
        // We must create a new map each run, as its constructor
        // allocates and initializes memory. This is the "build" cost.
        GpuHashMap<KeyT, ValueT> map(num_buckets, device_idx, seed, false);

        timer.start();
        map.buildTable(d_keys, d_values, num_keys);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // CRITICAL
        total_gpu_time += timer.elapsed_ms();

        if (i == 0) {
            // Get count once for validation
            result_gpu = map.countTable(); 
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
        // map object is destroyed here, freeing its memory
    }

    CHECK_CUDA_ERROR(cudaFree(d_keys));
    CHECK_CUDA_ERROR(cudaFree(d_values));

    double avg_gpu = total_gpu_time / num_runs;
    double throughput_gpu = (num_keys / (avg_gpu / 1000.0)) / 1e6; // Million Keys/sec

    std::cout << "Count: " << result_gpu << std::endl;
    std::cout << "Time:  " << std::fixed << std::setprecision(4)
              << avg_gpu << " ms" << std::endl;
    std::cout << "Perf:  " << std::fixed << std::setprecision(2)
              << throughput_gpu << " MK/s" << std::endl;

    // --- 4. Comparison ---
    std::cout << "\n--- PERFORMANCE ---" << std::endl;
    double speedup = avg_cpu / avg_gpu;
    std::cout << "Speedup (GPU vs CPU): " << std::fixed << std::setprecision(2)
              << speedup << "x" << std::endl;
    std::cout << "Throughput (GPU):     " << throughput_gpu << " MK/s" << std::endl;
    std::cout << "Throughput (CPU):     " << throughput_cpu << " MK/s" << std::endl;

    if (result_cpu != unique_key_count || result_gpu != unique_key_count) {
        std::cout << "\n⚠ WARNING: Results don't match expected unique count!" << std::endl;
        std::cout << "  Expected: " << unique_key_count << std::endl;
        std::cout << "  Got (CPU): " << result_cpu << std::endl;
        std::cout << "  Got (GPU): " << result_gpu << std::endl;
    } else {
        std::cout << "\n✓ Results verified correct" << std::endl;
    }
}

int main() {
    std::cout << "======== Insert Performance Comparison (GPU vs. CPU) ========" << std::endl;

    // 50% Load Factor Tests
    test_insert_performance(1000, 2000, false); // 1000 Sequential
    // test_insert_performance(1000, 2000, true); // random
    
    test_insert_performance(10000, 20000, false); // 10000 Sequential
    // test_insert_performance(10000, 20000, true); // random
    
    test_insert_performance(100000, 200000, false); // 100000 Sequential
    // test_insert_performance(100000, 200000, true); // random

    test_insert_performance(1000000, 2000000, false); // 1000000 Sequential
    // test_insert_performance(1000000, 2000000, true); // random

    test_insert_performance(10000000, 20000000, false); // 10000000 Sequential
    // test_insert_performance(10000000, 20000000, true); // random

    test_insert_performance(100000000, 200000000, false); // 100000000 Sequential
    // test_insert_performance(100000000, 200000000, true); // random

    std::cout << "\n======== Tests completed! ========" << std::endl;

    return 0;
}