include <iostream>
include <cuda_runtime.h>
include <cmath>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int Ns[] = {1000, 100000, 10000000};
    int blockSizes[] = {32, 128, 256, 512};

    for (int ni = 0; ni < 3; ni++) {
        int n = Ns[ni];
        size_t size = n * sizeof(float);

        float *h_a = new float[n];
        float *h_b = new float[n];
        float *h_c = new float[n];

        for (int i = 0; i < n; i++) {
            h_a[i] = i * 1.0f;
            h_b[i] = 2.0f * i;
        }

        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);

        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

        std::cout << "\nInput size n = " << n << std::endl;

        for (int bi = 0; bi < 4; bi++) {
            int blockSize = blockSizes[bi];
            int gridSize = (n + blockSize - 1) / blockSize;

            vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
            cudaDeviceSynchronize();

            cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

            bool correct = true;
            for (int i = 0; i < n; i++) {
                float expected = h_a[i] + h_b[i];
                if (fabs(h_c[i] - expected) > 1e-5) {
                    correct = false;
                    break;
                }
            }

            std::cout << "  blockSize = " << blockSize
                      << ", gridSize = " << gridSize
                      << ", totalThreads = " << gridSize * blockSize
                      << " -> " << (correct ? "PASS" : "FAIL")
                      << std::endl;
        }

        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    return 0;
}
