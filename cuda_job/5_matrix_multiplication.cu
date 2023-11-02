/* ---------------------------------------------------
  My Hello world for CUDA programming
  cd "/content/drive/Othercomputers/My MacBook Pro/cuda_job"
  --------------------------------------------------- */

#include <stdio.h>        // C programming header file
#include <unistd.h>       // C programming header file
// cude.h is automatically included by nvcc...
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>

#define MAXERR 1e-6

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct
{
    int width;
    int height;
    int *elements;
} Matrix;


/* ------------------------------------
   GPU Kernel (= GPU Function)
   ------------------------------------ */
// Get A element of (row, col)
__device__ int getElement(Matrix *A, int row, int col)
{
    return A->elements[row * A->width + col];
}
int gElement(Matrix *A, int row, int col)
{
    return A->elements[row * A->width + col];
}
// Set A element value of (row, col)
__device__ void setElement(Matrix *A, int row, int col, int value)
{
    A->elements[row * A->width + col] = value;
}
void deviceCheck()
{
    cudaError_t result = cudaSuccess;
    cudaDeviceSynchronize();
    result = cudaGetLastError();
    fprintf(stderr,"result=%d\n",result); fflush(stderr);
}

__global__ void matMulKernel(Matrix *m, Matrix *n, Matrix *p, int N)
{
    // Calculate row and column
    printf("blockIdx.x, blockIdx.y: %d, %d\n", blockIdx.x, blockIdx.y);
    printf("blockDim.x, blockDim.y: %d, %d\n", blockDim.x, blockDim.y);
    printf("threadIdx.x, threadIdx.y: %d, %d\n", threadIdx.x, threadIdx.y);
    int Cvalue = 10;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    printf("row: %d, column: %d\n", row, col);
    for (int i = 0; i < n->width; i++)
    {
        // &p->elements[row * n->width + i] = 20;
        Cvalue += getElement(m, row, i) * getElement(n, i, col);
        // printf("%d", Cvalue);
    }

    setElement(p, row, col, Cvalue);
    __syncthreads();
    // return 0;
}

void matMul_seqKernel(Matrix *m, Matrix *n, Matrix *p, int N, int height)
{
    for(int mrow = 0; mrow < height; mrow++) // M row
    {
        for(int ncolumn = 0; ncolumn < N; ncolumn++) // N column
        {
            for (int mcolumn = 0; mcolumn < N; mcolumn++) // M column
            {
                p->elements[mrow * N + ncolumn] += m->elements[mrow * N + mcolumn] * n->elements[mcolumn * N + ncolumn];
            }
        }
    }
}
int main(int argc, char *argv[])
{
    // Number of elements
    int N = atoi(argv[1]); //4; //1 << 20 2^20
    int height = atoi(argv[2]); //3;
    // Host pointers
    Matrix *h_m     = (Matrix *) malloc(sizeof(Matrix));
    Matrix *h_n     = (Matrix *) malloc(sizeof(Matrix));
    Matrix *h_p     = (Matrix *) malloc(sizeof(Matrix));
    Matrix *h_p_seq = (Matrix *) malloc(sizeof(Matrix));

    // Device pointers
    Matrix *d_m;
    Matrix *d_n;
    Matrix *d_p;

    size_t bytes = N * N * sizeof(int);
    size_t bytes2 = N * height * sizeof(int);
    // Allocate host memory
    // h_m->width = int(*) malloc(sizeof(int));
    // h_m->height = int(*) malloc(sizeof(int));
    h_m->elements     = (int *)malloc(bytes2);
    // h_n->width = int(*) malloc(sizeof(int));
    // h_n->height = int(*) malloc(sizeof(int));
    h_n->elements     = (int *)malloc(bytes2);
    // h_p->width = int(*) malloc(sizeof(int));
    // h_p->height = int(*) malloc(sizeof(int));
    h_p->elements     = (int *)malloc(bytes);
    // h_p_seq->width = int(*) malloc(sizeof(int));
    // h_p_seq->height = int(*) malloc(sizeof(int));
    h_p_seq->elements = (int *)malloc(bytes);


    // Initialize matrix m,n,p
    h_m->width = N;
    h_m->height = height;
    h_n->width = height;
    h_n->height = N;
    h_p->width = N;
    h_p->height = N;
    h_p_seq->width = N;
    h_p_seq->height = N;
    for (int i = 0; i < height; i++)
    {
        for(int j = 0; j < N; j++)
        {
            h_m->elements[i * height + j] = rand() % 5;
        }
    }
    // Initialize matrix m,n,p
    for (int i = 0; i < N; i++)
    {
        for(int j = 0; j < height; j++)
        {
            h_n->elements[i * N + j] = rand() % 5;
        }
    }
    // Initialize matrix m,n,p
    for (int i = 0; i < height; i++)
    {
        for(int j = 0; j < height; j++)
        {
            h_p_seq->elements[i * height + j] = 10;
        }
    }

    // Demo matrix m,n,p
    printf("\nMat m: \n");
    for (int i = 0; i < height; i++)
    {
        for(int j = 0; j < N; j++)
        {
            printf("%d ", gElement(h_m, i, j));
        }
        printf("\n");
    }
    printf("\nMat n: \n");
    for (int i = 0; i < N; i++)
    {
        for(int j = 0; j < height; j++)
        {
            printf("%d ", gElement(h_n, i, j));
        }
        printf("\n");
    }
    // Run in CPU to check
    matMul_seqKernel(h_m, h_n, h_p_seq, N, height);
    printf("\nMat p_seq: \n");
    for (int i = 0; i < height; i++)
    {
        for(int j = 0; j < height; j++)
        {
            printf("%d ", gElement(h_p_seq, i, j));
        }
        printf("\n");
    }
    printf("\n\ndone");
    cudaMallocManaged((Matrix **)&d_m, sizeof(Matrix));
    cudaMemcpy(d_m, h_m, sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMallocManaged((Matrix **)&d_n, sizeof(Matrix));
    cudaMemcpy(d_n, h_n, sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMallocManaged((Matrix **)&d_p, sizeof(Matrix));
    cudaMemcpy(d_p, h_p_seq, sizeof(Matrix), cudaMemcpyHostToDevice);

    printf("d2");
    // cudaMalloc((Matrix **)&d_m, sizeof(Matrix));
    // cudaMalloc((Matrix **)&d_n, sizeof(Matrix));
    // cudaMalloc((Matrix **)&d_p, sizeof(Matrix));
    // cudaMalloc((int **) &d_m->width,sizeof(int));
    // cudaMalloc((int **) &d_n->width,sizeof(int));
    // cudaMalloc((int **) &d_p->width,sizeof(int));
    // cudaMalloc((int **) &d_m->height,sizeof(int));
    // cudaMalloc((int **) &d_n->height,sizeof(int));
    // cudaMalloc((int **) &d_p->height,sizeof(int));

    // cudaMalloc((int **) &d_n->elements,bytes2);
    // cudaMalloc((int **) &d_p->elements,bytes);

    // Transfer data to device buff
    printf("done");
    // cudaMemcpy(d_m->elements, h_m->elements, bytes2, cudaMemcpyHostToDevice);
    // cudaMemcpy(&d_m->width, &h_m->width, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(&d_m->height, &h_m->height, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_n->elements, h_n->elements, bytes2, cudaMemcpyHostToDevice);
    // cudaMemcpy(&d_n->width, &h_n->width, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(&d_n->height, &h_n->height, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_p->elements, h_p_seq->elements, bytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(&d_p->width, &h_p->width, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(&d_p->height, &h_p->height, sizeof(int), cudaMemcpyHostToDevice);

    // cudaMemcpy(d_m, h_m, sizeof(Matrix), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_n, h_n, sizeof(Matrix), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_p, h_p_seq, sizeof(Matrix), cudaMemcpyHostToDevice);


    /* ------------------------------------
       Call Kernel Function
       ------------------------------------ */
    // initial block and grid size
    int threads_per_block = 4;
    dim3 block_size(threads_per_block, threads_per_block);
    dim3 grid_size((int)ceil( N / threads_per_block), (int)ceil( height / threads_per_block));
    printf("Block size is: (%d, %d).\n", block_size.x, block_size.y);
    printf("Grid size is: (%d, %d).\n", grid_size.x, grid_size.y);
    // Executing kernel
    matMulKernel <<< block_size, grid_size >>>(d_m, d_n, d_p, N);deviceCheck();
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    // Transfer data back to host memory
    cudaMemcpy(h_p, d_p, sizeof(Matrix), cudaMemcpyDeviceToHost);

    // Demo matrix m,n,p
    printf("\nMat p: \n");
    for (int i = 0; i < height; i++)
    {
        for(int j = 0; j < height; j++)
        {
            printf("%d ", gElement(h_p, i, j));
        }
        printf("\n");
    }


    // Deallocate device memory
    cudaFree(d_p);

    // Deallocate host memory

    return 0;
}
