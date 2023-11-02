#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 2048
#define MAX_ERR 10

__global__ void vector_add(float *out, float *a, float *b, int n) {
    // Calculate Index Thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n){
        out[tid] = a[tid] + b[tid];
    }
}

int main(){
    // Number of elements
    // int N = 100;

    // Host pointers
    float *a, *b, *out;

    // Device pointers
    float *d_a, *d_b, *d_out; 

    size_t bytes = sizeof(float) * N;
    // Allocate host memory
    a   = (float*)malloc(bytes);
    b   = (float*)malloc(bytes);
    out = (float*)malloc(bytes);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 0.1f;
        b[i] = 0.9f;
    }
    printf("%f, %f\n", a[N-1], b[N-1]);
    // Allocate device memory
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_out, bytes);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    // initial block and grid size
    int block_size = 1024;
    int grid_size = (int)ceil( (float) N / block_size);
    printf("Grid size is %d\n", grid_size);
    
    // Executing kernel 
    vector_add<<<grid_size,block_size>>>(d_out, d_a, d_b, N);
    
    // Transfer data back to host memory
    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) <= MAX_ERR);
    }
    printf("out[2] = %f\n", out[2047]);
    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}

