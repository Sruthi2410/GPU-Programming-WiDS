%%writefile shared_memory.cu
#include <iostream>
#include <cuda_runtime.h>

__global__ void global_kernel(float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < n-1)
        y[idx] = x[idx-1] + x[idx] + x[idx+1];
}

__global__ void shared_kernel(float* x, float* y, int n) {
    extern __shared__ float s[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    if (idx < n) s[t] = x[idx];
    __syncthreads();

    if (t > 0 && t < blockDim.x-1 && idx < n-1)
        y[idx] = s[t-1] + s[t] + s[t+1];
}

int main() {
    int n = 1 << 24;
    size_t size = n * sizeof(float);

    float *h_x = new float[n];
    for (int i = 0; i < n; i++) h_x[i] = i;

    float *d_x, *d_y;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    cudaEventRecord(s);
    global_kernel<<<gridSize, blockSize>>>(d_x, d_y, n);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float t1;
    cudaEventElapsedTime(&t1, s, e);

    cudaEventRecord(s);
    shared_kernel<<<gridSize, blockSize, blockSize*sizeof(float)>>>(d_x, d_y, n);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float t2;
    cudaEventElapsedTime(&t2, s, e);

    std::cout << "Global memory time: " << t1 << " ms\n";
    std::cout << "Shared memory time: " << t2 << " ms\n";
    std::cout << "Speedup: " << t1/t2 << "x\n";
}
