#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>

// Declaration of kernel wrappers defined in main.cu
extern __global__ void vector_add_kernel(const double* a, const double* b, double* result, int size);
extern __global__ void vector_scale_kernel(const double* a, double scalar, double* result, int dim);

// Test function for vector_add
void test_vector_add() {
    const int size = 5;
    double host_a[size]     = {1,  2,  3,  4,  5};
    double host_b[size]     = {5,  4,  3,  2,  1};
    double host_result[size] = {0};

    double *dev_a, *dev_b, *dev_result;
    cudaMalloc(&dev_a, size * sizeof(double));
    cudaMalloc(&dev_b, size * sizeof(double));
    cudaMalloc(&dev_result, size * sizeof(double));

    cudaMemcpy(dev_a, host_a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, size * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel on 1 thread/block (sufficient for our test)
    vector_add_kernel<<<1, 1>>>(dev_a, dev_b, dev_result, size);
    cudaMemcpy(host_result, dev_result, size * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        double expected = host_a[i] + host_b[i];
        assert(host_result[i] == expected);
    }
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);
    printf("test_vector_add passed.\n");
}

// Test function for vector_scale
void test_vector_scale() {
    const int dim = 5;
    double scalar = 2.0;
    double host_a[dim]       = {1, 2, 3, 4, 5};
    double host_result[dim]  = {0};

    double *dev_a, *dev_result;
    cudaMalloc(&dev_a, dim * sizeof(double));
    cudaMalloc(&dev_result, dim * sizeof(double));

    cudaMemcpy(dev_a, host_a, dim * sizeof(double), cudaMemcpyHostToDevice);
    vector_scale_kernel<<<1, 1>>>(dev_a, scalar, dev_result, dim);
    cudaMemcpy(host_result, dev_result, dim * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < dim; ++i) {
        double expected = host_a[i] * scalar;
        assert(host_result[i] == expected);
    }
    
    cudaFree(dev_a);
    cudaFree(dev_result);
    printf("test_vector_scale passed.\n");
}

int main() {
    test_vector_add();
    test_vector_scale();
    printf("All tests passed!\n");
    return 0;
}

