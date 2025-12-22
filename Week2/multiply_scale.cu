include <iostream>
include <cuda_runtime.h>
include <cmath>

__global__ void multiplyScale(const float* a, const float* b, float* c,
                              float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = alpha * a[idx] * b[idx];
    }
}

int main() {
    int n = 1000000;
    float alpha = 0.5f;
    size_t size = n * sizeof(float);

    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = 3.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    multiplyScale<<<gridSize, blockSize>>>(d_a, d_b, d_c, alpha, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < n; i++) {
        float expected = alpha * h_a[i] * h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }

    std::cout << "Multiply Scale: " << (correct ? "PASS" : "FAIL") << std::endl;

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
