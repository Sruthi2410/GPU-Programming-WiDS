import numpy as np
import time

H = 2048
W = 2048

image = np.random.rand(H, W).astype(np.float32)

kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], dtype=np.float32)

output = np.zeros((H, W), dtype=np.float32)

start = time.perf_counter()

for i in range(1, H - 1):
    for j in range(1, W - 1):
        s = 0.0
        for ki in range(-1, 2):
            for kj in range(-1, 2):
                s += kernel[ki + 1, kj + 1] * image[i + ki, j + kj]
        output[i, j] = s

end = time.perf_counter()

print("CPU 2D convolution time:", end - start, "seconds")
