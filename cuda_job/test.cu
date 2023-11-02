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
    float *elements;
} Matrix;


/* ------------------------------------
   GPU Kernel (= GPU Function)
   ------------------------------------ */
// Set A element value of (row, col)
__device__ void setElement(Matrix *A, int row, int col, int value)
{
    A->elements[row * A->width + col] = value;
}
__global__ void matMulKernel(Matrix *m, Matrix *n, Matrix *p)
{
    // Calculate row and column
    // printf("blockIdx.x, blockIdx.y: %d, %d\n", blockIdx.x, blockIdx.y);
    // printf("blockDim.x, blockDim.y: %d, %d\n", blockDim.x, blockDim.y);
    // printf("threadIdx.x, threadIdx.y: %d, %d\n", threadIdx.x, threadIdx.y);
    float sum = 0.0;
    int TILE_DIM = 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row > m->height || col > n->width)
        return;
    // printf("%d\n", col);
    // printf("%d\n", row);
    printf("row: %d, col: %d\n", row, col);
    int N = 1024;
    if(col <= N && row <= N)
    {
        for (int k = 0; k < n->width; k++)
        {
            sum += m->elements[row * n->width + k] * n->elements[k * n->width + col];
        }
    setElement(p, row, col, sum);
    }
    __syncthreads();

}

int main(int argc, char *argv[])
{
    // Number of elements
    int mcolumn = atoi(argv[1]); //4; //1 << 20 2^20
    int mrow = atoi(argv[2]); //3;
    int ncolumn = atoi(argv[3]);

    size_t mbytes = mcolumn * mrow * sizeof(float);
    size_t nbytes = mcolumn * ncolumn * sizeof(float);
    size_t pbytes = mrow * ncolumn * sizeof(float);
    // Host pointers
    Matrix *h_m = (Matrix *) malloc(sizeof(Matrix));
    Matrix *h_n = (Matrix *) malloc(sizeof(Matrix));
    Matrix *h_p = (Matrix *) malloc(sizeof(Matrix));

    // Device pointers
    Matrix *d_m;
    Matrix *d_n;
    Matrix *d_p;

    // Allocate host memory
    h_m->elements = (float *) malloc(mbytes);
    h_n->elements = (float *) malloc(nbytes);
    h_p->elements = (float *) malloc(pbytes);

    // Initialize matrix m,n,p
    h_m->width = mcolumn;
    h_m->height = mrow;
    h_n->width = ncolumn;
    h_n->height = mcolumn;
    h_p->width = ncolumn;
    h_p->height = mrow;

    printf("\nDDD\n");
    for (int i = 0; i < h_m->height; i++)
    {
        for(int j = 0; j < h_m->width; j++)
        {
            h_m->elements[i * h_m->width + j] = rand() % 5;
        }
    }
    printf("\nDDD\n");
    // Demo matrix m
    printf("\nMat m: \n");
    for (int i = 0; i < h_m->height; i++)
    {
        for(int j = 0; j < h_m->width; j++)
        {
            printf("%f ", h_m->elements[i * h_n->width + j]);
        }
        printf("\n");
    }
    printf("\nDDD\n");
    for (int i = 0; i < h_n->height; i++)
    {
        for(int j = 0; j < h_n->width; j++)
        {
            h_n->elements[i * h_n->width + j] = rand() % 5;
        }
    }
    printf("\nDDD\n");
    // Demo matrix n
    printf("\nMat n: \n");
    for (int i = 0; i < h_n->height; i++)
    {
        for(int j = 0; j < h_n->width; j++)
        {
            printf("%f ", h_n->elements[i * h_n->width + j]);
        }
        printf("\n");
    }
    printf("\nDDD\n");
    // for (int i = 0; i < h_p->height; i++)
    // {
    //     for(int j = 0; j < h_p->width; j++)
    //     {
    //         h_p->elements[i*h_p->width+j] = 3.0;
    //     }
    // }
    // Allocate device buff

    float *d_m_elementes = (float *) malloc(mbytes);
    float *d_n_elementes = (float *) malloc(nbytes);
    float *d_p_elementes = (float *) malloc(pbytes);
    cudaMallocManaged((float **) &d_m_elementes, mbytes);
    cudaMallocManaged((float **) &d_n_elementes, nbytes);
    cudaMallocManaged((float **) &d_p_elementes, pbytes);
    // cudaMallocManaged((Matrix **) &d_n, sizeof(Matrix));
    // cudaMallocManaged((Matrix **) &d_p, sizeof(Matrix));


    // Transfer data to device buff
    cudaMemcpy(d_m_elementes, h_m->elements, mbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_elementes, h_n->elements, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_elementes, h_p->elements, pbytes, cudaMemcpyHostToDevice);

    /* ------------------------------------
        Call Kernel Function
        ------------------------------------ */
    // initial block and grid size
    // int total_threads = mrow * ncolumn;
    int threads_per_block = 1;
    dim3 block_size(threads_per_block, threads_per_block);
    dim3 grid_size((int)ceil( ncolumn / threads_per_block), (int)ceil( mrow / threads_per_block));
    printf("Block size is: (%d, %d).\n", block_size.x, block_size.y);
    printf("Grid size is: (%d, %d).\n", grid_size.x, grid_size.y);
    // Executing kernel
    // matMulKernel <<< block_size, grid_size >>>(d_m, d_n, d_p);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    // Transfer data back to host memory
    cudaMemcpy(h_p->elements, d_m_elementes, pbytes, cudaMemcpyDeviceToHost);

    // Demo matrix m,n,p
    printf("\nMat p: \n");
    for (int i = 0; i < h_p->height; i++)
    {
        for(int j = 0; j < h_p->width; j++)
        {
            printf("%f ", h_p->elements[i * h_p->width + j]);
        }
        printf("\n");
    }
    printf("\nMatMul Finished.\n");

    // cudaFree(d_p);
    free(h_p);
    return 0;
}
