# GPU Hash Map Library
**GPU Course Project - Team 35**

A high-performance, warp-cooperative hash map implementation for CUDA GPUs. This library provides a generic key-value data structure designed for GPU applications, featuring concurrent operations, bulk processing capabilities, and adaptive search strategies.

## Project Overview

This project implements a complete GPU hash map library with the following features:
- **Warp-cooperative operations**: 32 threads work together for efficient hash table operations
- **Generic design**: Template-based to support various key and value types
- **Concurrent operations**: Support for mixed insert/delete/search batches
- **Hybrid search strategy**: Adaptive algorithm that switches between one-warp-per-key and one-thread-per-key based on workload size
- **Optimized count and delte operations**: Block-level reduction to minimize atomic operations
- **Iterator support**: Sequential traversal of all key-value pairs

Inspired by the SlabHash architecture (Ashkiani et al., IPDPS'18), this implementation adopts the warp-cooperative design philosophy but uses a simplified fixed-size table with linear probing instead of dynamic slab allocation. This design choice makes it efficient for applications with predictable table sizes while being easier to understand and extend.

## Build Instructions

### Prerequisites
- CUDA Toolkit 12.2 or higher
- CMake 3.18 or higher
- C++11 compatible compiler
- NVIDIA GPU with compute capability 3.5+

### Configure GPU Compute Capability

Edit line 4 of `CMakeLists.txt` to match your GPU architecture:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 75)  # Change to your GPU's compute capability
```

Common values:
- `75` - RTX 2080, RTX 2080 Ti (Turing)
- `80` - A100 (Ampere)
- `86` - RTX 3090 (Ampere)

### Build

```bash
mkdir build && cd build
cmake ..
make
```

Executables will be in `build/bin/`:
- `example` - Demonstration program
- `test_insert` - Insert operation tests
- `test_delete` - Delete operation tests
- `test_count_only` - Count operation tests
- `test_hash_map_comprehensive` - Comprehensive functionality tests
- `test_count_comprehensive` - Comprehensive count tests
- `test_count_performance` - Count performance benchmarks
- `test_hybrid_search` - Hybrid search strategy tests
- `test_iterator` - Iterator functionality tests
- `test_debug` - Debug utilities
- `test_insert_performance` - insert performance tests
- `test_delete_performance` - delete performance tests
- `test_search_performance` - search performance tests

## Usage

### Basic Example

```cuda
#include "gpu_hash_map.cuh"

// Create hash map with 1M buckets on GPU 0
GpuHashMap<uint32_t, uint32_t> hash_map(1000000, 0);

// Prepare data on GPU
uint32_t num_keys = 100000;
uint32_t* d_keys;
uint32_t* d_values;
cudaMalloc(&d_keys, num_keys * sizeof(uint32_t));
cudaMalloc(&d_values, num_keys * sizeof(uint32_t));

// Insert key-value pairs
hash_map.buildTable(d_keys, d_values, num_keys);

// Search for keys
uint32_t* d_results;
cudaMalloc(&d_results, num_keys * sizeof(uint32_t));
hash_map.searchTable(d_keys, d_results, num_keys);

// Delete keys
hash_map.deleteTable(d_keys, num_keys);

// Count remaining entries
uint32_t count = hash_map.countTable();

// Cleanup
cudaFree(d_keys);
cudaFree(d_values);
cudaFree(d_results);
```

### Iterator Usage

```cuda
// Traverse all key-value pairs
auto iter = hash_map.getIterator();
while (iter.hasNext()) {
  auto pair = iter.next();
  std::cout << pair.key << " -> " << pair.value << std::endl;
}
```

### Warp-Cooperative Operations in Custom Kernels

```cuda
#include "gpu_hash_map.cuh"

__global__ void custom_kernel(GpuHashMapContext<uint32_t, uint32_t> ctx,
                               uint32_t* keys, uint32_t* values, uint32_t num) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num) return;

  // Each thread may have work
  bool has_work = (tid < num);
  uint32_t my_key = has_work ? keys[tid] : 0;
  uint32_t my_value = has_work ? values[tid] : 0;

  // Compute bucket
  uint32_t bucket = ctx.computeBucket(my_key);

  // Warp cooperates to insert
  ctx.insertKey(has_work, laneId, my_key, my_value, bucket);
}
```

## API Reference

### GpuHashMap<KeyT, ValueT>

Host-side class that owns GPU memory.

**Constructor:**
```cpp
GpuHashMap(uint32_t num_buckets,
           uint32_t device_idx = 0,
           int64_t seed = 0,
           bool verbose = false,
           uint32_t search_warp_threshold = 5000)
```

**Parameters:**
- `num_buckets` - Size of hash table
- `device_idx` - GPU device to use (default: 0)
- `seed` - Random seed for hash function (default: 0)
- `verbose` - Print debug information (default: false)
- `search_warp_threshold` - Queries below this use one-warp-per-key, above use one-thread-per-key (default: 5000)

**Operations:**
- `void buildTable(KeyT* d_keys, ValueT* d_values, uint32_t num_keys)` - Bulk insert
- `void searchTable(KeyT* d_queries, ValueT* d_results, uint32_t num_queries)` - Bulk search (hybrid strategy)
- `void deleteTable(KeyT* d_keys, uint32_t num_keys)` - Bulk delete
- `void deleteTableOptimized(KeyT* d_keys, uint32_t num_keys)` - Optimized bulk delete with reduced atomic operations
- `uint32_t countTable()` - Count valid elements
- `uint32_t countTableOptimized()` - Optimized count with block-level reduction
- `void clear()` - Clear all entries
- `GpuHashMapContext<KeyT, ValueT> getContext()` - Get device context for custom kernels
- `GpuHashMapIterator<KeyT, ValueT> getIterator()` - Get iterator for sequential traversal

### GpuHashMapContext<KeyT, ValueT>

Device-side class for use in kernels (shallow-copied, does not own memory).

**Operations:**
- `__device__ uint32_t computeBucket(const KeyT& key)` - Hash function
- `__device__ void insertKey(bool, uint32_t laneId, KeyT, ValueT, uint32_t bucket)` - Warp-cooperative insert
- `__device__ void searchKey(bool, uint32_t laneId, KeyT, ValueT&, uint32_t bucket)` - Warp-cooperative search
- `__device__ void deleteKey(bool, uint32_t laneId, KeyT, uint32_t bucket)` - Warp-cooperative delete
- `__device__ void countKey(bool, uint32_t laneId, KeyT, uint32_t&, uint32_t bucket)` - Warp-cooperative count

### GpuHashMapIterator<KeyT, ValueT>

Iterator for sequential traversal of all key-value pairs.

**Operations:**
- `bool hasNext()` - Check if more entries exist
- `KeyValuePair<KeyT, ValueT> next()` - Get next key-value pair

## Testing

```bash
cd build/bin

# Check CUDA version and basic functionality
./test_cuda_version

# Run optimized delete tests
./test_delete_optimized

# Run basic correctness tests
./test_basic

# Run concurrent operations tests
./test_concurrent

# Run example demonstration
./example

# Correctness tests
./test_insert                      # Insert operation tests
./test_delete                      # Delete operation tests
./test_count_only                  # Count operation tests
./test_hash_map_comprehensive      # Comprehensive functionality tests
./test_count_comprehensive         # Comprehensive count tests
./test_iterator                    # Iterator functionality tests
./test_hybrid_search               # Hybrid search strategy tests

# Performance tests
./test_count_performance           # Count performance benchmarks
./test_serach_performance          # Search performance test
./test_insert_performance          # Insert performance test
./test_delete_performance          # delete performance test

# Debug utilities
./test_debug
```

## Implementation Details & Performance

### Collision Resolution
- **Strategy**: Linear probing with open addressing
- **Maximum probe distance**: 128 slots
- **Slot states**: EMPTY, OCCUPIED, TOMBSTONE, PENDING

### Hash Function
Universal hashing with prime modulo:
```
bucket = (((hash_x ^ key) + hash_y) % PRIME_DIVISOR) % num_buckets
PRIME_DIVISOR = 4294967291u
```

### Hybrid Search Strategy
The library automatically adapts search strategy based on workload size:
- **Small workloads** (< threshold, default 5000): One warp per key
- **Large workloads** (>= threshold): One thread per key
- Threshold configurable via constructor parameter

### Performance Characteristics
- **Warp efficiency**: Operations are most efficient when threads in a warp access nearby buckets
- **Load factor**: Performance degrades with high load factors (>0.7). Recommended: 0.5-0.6
- **Hash quality**: Good hash function distribution is critical for performance
- **Insert protocol**: Three-state protocol with READ-FIRST optimization reduces contention
- **Count and Delete optimization**: Block-level reduction minimizes atomic operations compared to naive implementation

## Project Structure

```
GPU-Team35-2025/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── CLAUDE.md                   # AI coding assistant guidance
├── include/
│   └── gpu_hash_map.cuh       # Main public API (single include file)
├── src/
│   ├── hash_map_impl.cuh      # Host-side class (owns memory)
│   ├── hash_map_context.cuh   # Device-side context (no ownership)
│   ├── iterator.cuh           # Iterator support
│   ├── warp/                  # Warp-cooperative operations
│   │   ├── insert.cuh         # Insert operation
│   │   ├── search.cuh         # Search operation
│   │   ├── delete.cuh         # Delete operation
│   │   └── count.cuh          # Count operation
│   └── kernels/               # CUDA kernel implementations
│       ├── build_kernel.cuh           # Bulk insert
│       ├── search_kernel.cuh          # Bulk search (hybrid strategy)
│       ├── delete_kernel.cuh          # Bulk delete
│       ├── count_kernel.cuh           # Count entries
│       ├── count_kernel_optimized.cuh # Optimized count
│       └── dump_kernels.cuh           # Debug utilities
├── examples/
│   └── main.cu                # Usage demonstration
└── test/
    ├── test_utils.cuh                 # Testing utilities
    ├── test_insert.cu                 # Insert tests
    ├── test_delete.cu                 # Delete tests
    ├── test_delete_performance.cu     # Delete performance tes
    ├── test_count_only.cu             # Count tests
    ├── test_hash_map_comprehensive.cu # Comprehensive tests
    ├── test_count_comprehensive.cu    # Count comprehensive tests
    ├── test_count_performance.cu      # Count benchmarks
    ├── test_hybrid_search.cu          # Hybrid search tests
    ├── test_search_performance.cu     # Search Perfromance
    ├── test_insert_performance.cu     # 
    ├── test_iterator.cu               # Iterator tests
    └── test_debug.cu                  # Debug tests
```

## References

- **SlabHash**: [Ashkiani et al., IPDPS'18](https://ieeexplore.ieee.org/abstract/document/8425196)
- **CUDA Programming Guide**: [NVIDIA Documentation](https://docs.nvidia.com/cuda/)

## License

MIT License - See LICENSE file for details.

## Authors

Team 35 - GPU Course Project 2025
