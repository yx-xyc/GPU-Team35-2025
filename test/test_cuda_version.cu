/*
 * CUDA Version Test
 *
 * This test verifies that CUDA 12.2 or higher is available and working correctly.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

using namespace std;

int main() {
    cout << "=== CUDA Version Test ===" << endl;
    
    // Get CUDA runtime version
    int runtime_version;
    cudaError_t error = cudaRuntimeGetVersion(&runtime_version);
    if (error != cudaSuccess) {
        cerr << "Failed to get CUDA runtime version: " << cudaGetErrorString(error) << endl;
        return 1;
    }
    
    // Get CUDA driver version
    int driver_version;
    error = cudaDriverGetVersion(&driver_version);
    if (error != cudaSuccess) {
        cerr << "Failed to get CUDA driver version: " << cudaGetErrorString(error) << endl;
        return 1;
    }
    
    // Parse versions
    int runtime_major = runtime_version / 1000;
    int runtime_minor = (runtime_version % 1000) / 10;
    
    int driver_major = driver_version / 1000;
    int driver_minor = (driver_version % 1000) / 10;
    
    cout << "CUDA Runtime Version: " << runtime_major << "." << runtime_minor << endl;
    cout << "CUDA Driver Version: " << driver_major << "." << driver_minor << endl;
    
    // Check if CUDA 12.2 or higher is available
    bool version_ok = (runtime_major > 12) || (runtime_major == 12 && runtime_minor >= 2);
    
    if (version_ok) {
        cout << "✓ CUDA version requirement met (12.2+)" << endl;
    } else {
        cout << "✗ CUDA version requirement not met (need 12.2+)" << endl;
        return 1;
    }
    
    // Get device properties
    int device_count;
    error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        cerr << "Failed to get device count: " << cudaGetErrorString(error) << endl;
        return 1;
    }
    
    cout << "Number of CUDA devices: " << device_count << endl;
    
    if (device_count == 0) {
        cerr << "No CUDA devices found!" << endl;
        return 1;
    }
    
    // Test device 0
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, 0);
    if (error != cudaSuccess) {
        cerr << "Failed to get device properties: " << cudaGetErrorString(error) << endl;
        return 1;
    }
    
    cout << "Device 0: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << endl;
    
    // Simple kernel test
    cout << "\nTesting basic CUDA functionality..." << endl;
    
    int* d_test;
    error = cudaMalloc(&d_test, sizeof(int));
    if (error != cudaSuccess) {
        cerr << "Failed to allocate device memory: " << cudaGetErrorString(error) << endl;
        return 1;
    }
    
    int h_test = 42;
    error = cudaMemcpy(d_test, &h_test, sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cerr << "Failed to copy to device: " << cudaGetErrorString(error) << endl;
        cudaFree(d_test);
        return 1;
    }
    
    int h_result = 0;
    error = cudaMemcpy(&h_result, d_test, sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cerr << "Failed to copy from device: " << cudaGetErrorString(error) << endl;
        cudaFree(d_test);
        return 1;
    }
    
    cudaFree(d_test);
    
    if (h_result == 42) {
        cout << "✓ Basic CUDA memory operations working" << endl;
    } else {
        cout << "✗ Basic CUDA memory operations failed" << endl;
        return 1;
    }
    
    cout << "\n✓ All CUDA tests passed!" << endl;
    return 0;
}
