# GPU Hash Map Library
**GPU Course Project - Team 35**

A high-performance, warp-cooperative hash map implementation for CUDA GPUs. This library provides a generic key-value data structure designed for GPU applications, featuring concurrent operations and bulk processing capabilities.

## Project Overview

This project implements a GPU hash map library with the following goals:
- **Warp-cooperative operations**: 32 threads work together for efficient hash table operations
- **Generic design**: Template-based to support various key and value types
- **Concurrent operations**: Support for mixed insert/delete/search batches
- **Simple starting point**: Fixed-size table design that can evolve to dynamic allocation

Inspired by the SlabHash architecture (Ashkiani et al., IPDPS'18), this implementation starts with a simpler fixed-size design suitable for learning and prototyping, with a clear path to more advanced features.

## Features

### Current Implementation
- ✅ **Core Operations**: Insert, Search, Delete, Count
- ✅ **Bulk Operations**: Batch insert, search, and delete
- ✅ **Concurrent Operations**: Mixed operation batches
- ✅ **Iterator Support**: Traverse all key-value pairs
- ✅ **Warp-Cooperative Design**: Efficient GPU utilization
- ✅ **Generic Templates**: Support for custom key/value types

### Planned Features (See TODO.md)
- 🔄 Dynamic memory allocation (SlabAlloc integration)
- 🔄 Chaining for collision resolution
- 🔄 Performance benchmarking framework
- 🔄 Advanced hash functions

## Build Instructions

### Prerequisites
- CUDA Toolkit 8.0 or higher
- CMake 3.8 or higher
- C++11 compatible compiler
- NVIDIA GPU with compute capability 3.5+

### Configure GPU Compute Capability

Edit `CMakeLists.txt` to match your GPU architecture. For example:
```cmake
option(HASHMAP_GENCODE_SM75 "GENCODE_SM75" ON)  # For RTX 2080 (Turing)
option(HASHMAP_GENCODE_SM80 "GENCODE_SM80" ON)  # For A100 (Ampere)
```

Or use `ccmake` for interactive configuration:
```bash
mkdir build && cd build
ccmake ..  # Toggle architecture options
```

### Build

```bash
mkdir build && cd build
cmake ..
make
```

Executables will be in `build/bin/`:
- `example` - Demonstration program
- `test_basic` - Basic correctness tests
- `test_concurrent` - Concurrent operations tests

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

// Cleanup
cudaFree(d_keys);
cudaFree(d_values);
cudaFree(d_results);
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
GpuHashMap(uint32_t num_buckets, uint32_t device_idx = 0,
           int64_t seed = 0, bool verbose = false)
```

**Operations:**
- `void buildTable(KeyT* d_keys, ValueT* d_values, uint32_t num_keys)` - Bulk insert
- `void searchTable(KeyT* d_queries, ValueT* d_results, uint32_t num_queries)` - Bulk search
- `void deleteTable(KeyT* d_keys, uint32_t num_keys)` - Bulk delete
- `uint32_t countTable()` - Count valid elements
- `void clear()` - Clear all entries
- `GpuHashMapContext<KeyT, ValueT> getContext()` - Get device context

### GpuHashMapContext<KeyT, ValueT>

Device-side class for use in kernels (shallow-copied).

**Operations:**
- `__device__ uint32_t computeBucket(const KeyT& key)` - Hash function
- `__device__ void insertKey(bool, uint32_t laneId, KeyT, ValueT, uint32_t bucket)` - Insert
- `__device__ void searchKey(bool, uint32_t laneId, KeyT, ValueT&, uint32_t bucket)` - Search
- `__device__ void deleteKey(bool, uint32_t laneId, KeyT, uint32_t bucket)` - Delete

## Architecture

```
┌─────────────────────────────────────┐
│     GpuHashMap (Host)               │
│  - Owns GPU memory                  │
│  - Manages lifecycle                │
│  - Launches kernels                 │
└──────────┬──────────────────────────┘
           │ Shallow copy
           ▼
┌─────────────────────────────────────┐
│  GpuHashMapContext (Device)         │
│  - No memory ownership              │
│  - Used in kernels                  │
│  - Warp-cooperative ops             │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│    Warp (32 threads)                │
│  - Cooperative operations           │
│  - Linear probing                   │
│  - Ballot/shuffle intrinsics        │
└─────────────────────────────────────┘
```

## Testing

```bash
cd build/bin

# Run basic correctness tests
./test_basic

# Run concurrent operations tests
./test_concurrent

# Run example demonstration
./example
```

## Performance Notes

- **Warp efficiency**: Operations are most efficient when threads in a warp access nearby buckets
- **Load factor**: Performance degrades with high load factors (>0.7). Recommended: 0.5-0.6
- **Hash quality**: Good hash function distribution is critical for performance
- **Concurrent operations**: Mixed operations may have lower throughput than pure operations

## Project Structure

```
GPU-Team35-2025/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── TODO.md                  # Future enhancements
├── include/
│   └── gpu_hash_map.cuh    # Main public API
├── src/
│   ├── hash_map_context.cuh   # Device context
│   ├── hash_map_impl.cuh      # Host implementation
│   ├── warp/                  # Warp operations
│   ├── kernels/               # CUDA kernels
│   └── iterator.cuh           # Iterator support
├── examples/
│   └── main.cu               # Usage examples
└── test/
    ├── test_basic.cu         # Correctness tests
    └── test_concurrent.cu    # Concurrent tests
```

## References

- **SlabHash**: [Ashkiani et al., IPDPS'18](https://ieeexplore.ieee.org/abstract/document/8425196)
- **CUDA Programming Guide**: [NVIDIA Documentation](https://docs.nvidia.com/cuda/)

## License

MIT License - See LICENSE file for details.

## Team Members

Team 35 - GPU Course Project 2025

## Contributing

This is a course project. For questions or suggestions, please coordinate with team members.
