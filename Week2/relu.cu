include <iostream>
include <cuda_runtime.h>
include <cmath>

__global__ void relu(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = (x[idx] > 0.0f) ? x[idx] : 0.0f;
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    float *h_x = new float[n];
    float *h_y = new float[n];

    for (int i = 0; i < n; i++) {
        h_x[i] = (i % 2 == 0) ? i * 1.0f : -i * 1.0f;
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    relu<<<gridSize, blockSize>>>(d_x, d_y, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < n; i++) {
        float expected = (h_x[i] > 0.0f) ? h_x[i] : 0.0f;
        if (fabs(h_y[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }

    std::cout << "ReLU: " << (correct ? "PASS" : "FAIL") << std::endl;

    delete[] h_x;
    delete[] h_y;
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
