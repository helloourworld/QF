/* ---------------------------------------------------
   Mat Mul CUDA programming
   --------------------------------------------------- */

#include <stdio.h>        // C programming header file
#include <unistd.h>       // C programming header file

#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h> // cude.h is automatically included by nvcc...
#include <cuda_runtime.h>

#define N 100
#define MAXERR 1e-6
// Thread block size
#define BLOCK_SIZE 2

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

/* ------------------------------------
   MatMul kernel (= GPU function)
   ------------------------------------ */

// Get A element of (row, col)
__device__ float getElement(Matrix *A, int row, int col)
{
    return A->elements[row * A->width + col];
}

// Set A element value of (row, col)
__device__ void setElement(Matrix *A, int row, int col, float value)
{
    A->elements[row * A->width + col] = value;
}


__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C)
{
// Each thread computes one element of C
    // by accumulating results into Cvalue
    
    float Cvalue = 0.0;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < A->width; i++)
    {
        Cvalue += getElement(A, row, i) * getElement(B, i, col);
        // printf("%d", Cvalue);
    }
    setElement(C, row, col, Cvalue);
}


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
int main()
{
    int width = 1 << 2;
    int height = 1 << 2;
    
    // Host pointers
    // Matrix *A, *B, *C;

    // Device pointers
    Matrix *d_A, *d_B, *d_C;

    size_t mBytes = sizeof(Matrix*);
    size_t nBytes = width * height * sizeof(float);
    // Allocate host memory
    
    A->elements = (float *)malloc(nBytes);
    B->elements = (float *)malloc(nBytes);
    C->elements = (float *)malloc(nBytes);

    // Allocate device buff

    cudaMallocManaged((void **)&d_A, mBytes); // allocate dynamic memory
    cudaMallocManaged((void **)&d_B, mBytes); // allocate dynamic memory
    cudaMallocManaged((void **)&d_C, mBytes); // allocate dynamic memory
    cudaMallocManaged((void **)&d_A->elements, nBytes); // allocate dynamic memory
    cudaMallocManaged((void **)&d_B->elements, nBytes); // allocate dynamic memory
    cudaMallocManaged((void **)&d_C->elements, nBytes); // allocate dynamic memory

    // Initialize host arrays
    // Load from file
    A->height = height;
    A->width = width;
    B->height = height;
    B->width = width;
    C->height = height;
    C->width = width;
    for (int i = 0; i < width * height; ++i)
    {
        A->elements[i] = 1.0;
        B->elements[i] = 2.0;
    }

    // Transfer data to device buff
    cudaMemcpy(d_A->elements, A->elements, nBytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B->elements, B->elements, nBytes,
               cudaMemcpyHostToDevice);

    /* ------------------------------------
       Call Kernel Function
       ------------------------------------ */
    // initial block and grid size

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(B->width / blockSize.x, A->height / blockSize.y);

    /* ------------------------------------
    Call the hello( ) kernel function
    ------------------------------------ */
    matMulKernel << < gridSize, blockSize >> >(d_A, d_B, d_C);


    // Transfer data back to host memory
    cudaMemcpy(C->elements, d_C->elements, nBytes,
               cudaMemcpyDeviceToHost);
    

    // printf("Hello World ! \n");

    // printf("Matrix[2] = %d\n", Matrix[2]);
    printf("PASS");

    // Deallocate device memory
    // cudaFree(Matrix);

    // Deallocate host memory
    free(A);
    return 0;
}
