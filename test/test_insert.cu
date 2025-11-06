#include <iostream>
#include <cuda_runtime.h>
#include "gpu_hash_map.cuh"
#include "../src/kernels/dump_kernels.cuh"

// TEST RESULTS !!!:
//  ./test_insert 
// === GPU Hash Map ===
//   Num buckets: 16
//   Device: 0
//   Hash parameters: (3243368313, 2744618934)
//   Memory: 0.000183105 MB

// Table after basic inserts:
//   slot  3: key=1, val=10
//   slot  5: key=3, val=30
//   slot  6: key=2, val=20
//   slot  8: key=4, val=40

void test_basic_insert() {
    const uint32_t buckets = 16;
    GpuHashMap<int, int> map(buckets, 0, 42, true);

    int h_keys[] = {1, 2, 3, 4};
    int h_vals[] = {10, 20, 30, 40};
    const uint32_t num_keys = 4;

    int *d_keys, *d_vals;
    cudaMalloc(&d_keys, sizeof(int) * num_keys);
    cudaMalloc(&d_vals, sizeof(int) * num_keys);
    cudaMemcpy(d_keys, h_keys, sizeof(h_keys), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals, sizeof(h_vals), cudaMemcpyHostToDevice);

    map.buildTable(d_keys, d_vals, num_keys);

    // Retrieve table contents
    GpuHashMapContext<int, int> ctx = map.getContext();
    int *d_out_keys, *d_out_vals;
    uint32_t *d_out_status;
    cudaMalloc(&d_out_keys, buckets * sizeof(int));
    cudaMalloc(&d_out_vals, buckets * sizeof(int));
    cudaMalloc(&d_out_status, buckets * sizeof(uint32_t));

    dump_table_kernel<<<1, buckets>>>(ctx, d_out_keys, d_out_vals, d_out_status);
    cudaDeviceSynchronize();

    int h_out_keys[buckets], h_out_vals[buckets];
    uint32_t h_out_status[buckets];
    cudaMemcpy(h_out_keys, d_out_keys, buckets * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_vals, d_out_vals, buckets * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_status, d_out_status, buckets * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Table after basic inserts:\n");
    for (int i = 0; i < buckets; ++i) {
        if (h_out_status[i] == OCCUPIED) {
            printf("  slot %2d: key=%d, val=%d\n", i, h_out_keys[i], h_out_vals[i]);
        }
    }

    cudaFree(d_keys);
    cudaFree(d_vals);
    cudaFree(d_out_keys);
    cudaFree(d_out_vals);
    cudaFree(d_out_status);
}

int main() {
    test_basic_insert();
    // test_collision();
    // test_duplicate_update();
    // test_near_full_table();
    return 0;
}
