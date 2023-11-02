#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>


//thread 1D
__global__ void addKernal(int * c, const int * a, const int * b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



int main()
{
    const int n = 5;

    const int a[n] = {1, 2, 3, 4, 5};
    const int b[n] = {10, 2, 30, 4, 50};
    int c[n]  = {0};

    int *d_a, *d_b, *d_c;

    // Allocate GPU buffer
    cudaMalloc((void**)&d_c, n * sizeof(int));
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_b, n * sizeof(int));
    
    // Copy input vectors from host memory to GPU buffer.
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, n * sizeof(int), cudaMemcpyHostToDevice);


    addKernal <<<1, n>>>(d_c, d_a, d_b);
    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

    printf("[1, 2, 3, 4, 5]\n + \n[10, 2, 30, 4, 50] \n = \n [%d, %d, %d, %d, %d]\n",
        c[0], c[1], c[2], c[3], c[4]);

    return 0;
}