%%writefile non_coalesced.cu
#include <iostream>
#include <cuda_runtime.h>

#define STRIDE 32

__global__ void non_coalesced(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int access = (idx * STRIDE) % n;  // scattered access
        c[access] = a[access] + b[access];
    }
}

int main() {
    int n = 1 << 24;
    size_t size = n * sizeof(float);

    float *h_a = new float[n];
    float *h_b = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Warm-up
    non_coalesced<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        non_coalesced<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Non-coalesced time (avg over 10 runs): "
              << ms / 10 << " ms\n";

    return 0;
}
