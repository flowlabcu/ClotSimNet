#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>

// Compile with g++ alloc.cc -o alloc.so -I/usr/local/cuda/include -shared -fPIC
extern "C" {
void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
    void *ptr;
    cudaMallocManaged(&ptr, size); // Use cudaMallocManaged for unified memory
    return ptr;
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    cudaFree(ptr);
}
}
