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
#include <errno.h>
#define BLOCK_SIZE 4

/* ------------------------------------
   GPU Kernel (= GPU Function)
   ------------------------------------ */
__global__ void matMulKernel(float *A, float *B, float *C, int m, int n, int k)
{
    // Calculate row and column
    // printf("blockIdx.x, blockIdx.y: %d, %d\n", blockIdx.x, blockIdx.y);
    // printf("blockDim.x, blockDim.y: %d, %d\n", blockDim.x, blockDim.y);
    // printf("threadIdx.x, threadIdx.y: %d, %d\n", threadIdx.x, threadIdx.y);
    float Cvalue = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col <= k && row <= m) 
    {
        for(int i = 0; i < n; i++) 
        {
            Cvalue += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = Cvalue;
    }
} 

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

void matMul_seqKernel(Matrix *m, Matrix *n, Matrix *p, int mwidth, int mheight, int nwidth)
{
    for(int mrow = 0; mrow < mheight; mrow++) // M row
    {
        for(int ncolumn = 0; ncolumn < nwidth; ncolumn++) // N column
        {
            for (int mcolumn = 0; mcolumn < mwidth; mcolumn++) // M column
            {
                p->elements[mrow * nwidth + ncolumn] += m->elements[mrow * nwidth + mcolumn] * n->elements[mcolumn * nwidth + ncolumn];
            }
        }
    }
}

float** load_matrix(size_t *_row_number, size_t *_column_number, const char *_file_path)
{

    *_row_number = 0;
    *_column_number = 0;

    /* open file */
    FILE *fp = fopen(_file_path, "r");
    if(fp == NULL)
    {
        fprintf(stderr, "Please check! No file %s, %s\n\n", _file_path, strerror(errno));
    }

    /* detect file content*/
    float **matrix=NULL, **tmp;
    char line[1024];

    /* read file */
    while(fgets(line, sizeof(line), fp))
    {
        /* determine columns size */
        if(*_column_number == 0)
        {
            char *scan = line;
            float dummy;
            int offset = 0;
            while(sscanf(scan, "%f%n", &dummy, &offset) == 1)
            {
                scan += offset;
                (*_column_number)++;
            }
        }

        tmp = (float **) realloc(matrix, (*_row_number + 1) * sizeof(*matrix)); // dynamic memory
        /* if no data */
        if(tmp == NULL)
        {
            fclose(fp);
            printf("d");
            return matrix;
        }
        matrix = tmp;
        matrix[*_row_number] = (float *) calloc(*_column_number, sizeof(*matrix[*_row_number])); // allocate dynamic memory
        if(matrix[*_row_number] == NULL)
        {
            fclose(fp);
            if(*_row_number == 0)
            {
                fclose(fp);
            }
        }
        /* load data */
        int offset = 0;
        char *scan = line;
        for(size_t j = 0; j < *_column_number; j++)
        {
            if(sscanf(scan, "%f%n", matrix[*_row_number] + j, &offset) == 1)
                scan += offset;
            else
                matrix[*_row_number][j] = 0.0; // Missing then set 0
        }

        // incrementing _row_number
        (*_row_number)++;
    }
    fclose(fp);
    return matrix;
}

int main(int argc, char *argv[])
{
    // Number of elements
    char *_file_path = argv[1];
    size_t mrow, mcolumn;

    // int mcolumn = atoi(argv[1]); //4; //1 << 20 2^20
    // int mrow = atoi(argv[2]); //3;
    // int ncolumn = atoi(argv[3]);

    // size_t mbytes = mcolumn * mrow * sizeof(float);
    // size_t nbytes = mcolumn * ncolumn * sizeof(float);
    // size_t pbytes = mrow * ncolumn * sizeof(float);
    // // Host pointers
    // Matrix *h_m = (Matrix *) malloc(sizeof(Matrix));
    // Matrix *h_n = (Matrix *) malloc(sizeof(Matrix));
    // Matrix *h_p = (Matrix *) malloc(sizeof(Matrix));
    // Matrix *h_p_seq = (Matrix *) malloc(sizeof(Matrix));

    // // Device pointers
    // float *dev_a;
    // float *dev_b;
    // float *dev_c;

    // // Allocate host memory
    // float *h_m_elements;h_m_elements = (float*) malloc(mbytes);
    // float *h_n_elements;h_n_elements = (float*) malloc(nbytes);
    // float *h_p_elements;h_p_elements = (float*) malloc(pbytes);
    // float *h_p_seq_elements;h_p_seq_elements = (float*) malloc(pbytes);

    // // Initialize matrix m,n,p
    // h_m->width = mcolumn;
    // h_m->height = mrow;
    // h_n->width = ncolumn;
    // h_n->height = mcolumn;
    // h_p->width = ncolumn;
    // h_p->height = mrow;
    // h_p_seq->width = ncolumn;
    // h_p_seq->height = mrow;
    // printf("\nDDD\n");
    // for (int i = 0; i < mrow; i++)
    // {
    //     for(int j = 0; j < mcolumn; j++)
    //     {
    //         h_m_elements[i * mcolumn + j] = rand() % 5;
    //     }
    // }
    // printf("\nDDD\n");
    
    //input in host array
    /* parameters check */

    float **m = load_matrix(&mrow, &mcolumn, argv[1]);

    // Demo matrix m
    printf("\nMat m: \n");
    for (int i = 0; i < mrow; i++)
    {
        for(int j = 0; j < mcolumn; j++)
        {
            printf("%f ", m[i][j]);
        }
        printf("\n");
    }
    printf("\nDDD\n");

    // for(int i = 0; i < _row_number; ++i)
    // {
    //     for(int j = 0; j < _column_number; ++j)
    //         printf("%f ", matrix[i][j]);
    //     puts("");

    // }
    // for (int i = 0; i < mcolumn; i++)
    // {
    //     for(int j = 0; j < ncolumn; j++)
    //     {
    //         h_n_elements[i * ncolumn + j] = rand() % 5;
    //     }
    // }
    // printf("\nDDD\n");
    // // Demo matrix n
    // printf("\nMat n: \n");
    // for (int i = 0; i < mcolumn; i++)
    // {
    //     for(int j = 0; j < ncolumn; j++)
    //     {
    //         printf("%f ", h_n_elements[i * ncolumn + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\nDDD\n");
    // // Run in CPU to check
    // h_m->elements = h_m_elements;
    // h_n->elements = h_n_elements;
    // h_p_seq->elements = h_p_seq_elements;
    
    // matMul_seqKernel(h_m, h_n, h_p_seq, mcolumn, mrow, ncolumn);

    // printf("\nMat p_seq: \n");
    // for (int i = 0; i < mrow; i++)
    // {
    //     for(int j = 0; j < ncolumn; j++)
    //     {
    //         printf("%f ", h_p_seq->elements[i * ncolumn + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n\ndone");
    // // Allocate device buff

    // cudaMalloc((float **) &dev_a, mbytes);
    // cudaMalloc((float **) &dev_b, nbytes);
    // cudaMalloc((float **) &dev_c, pbytes);


    // // Transfer data to device buff
    // cudaMemcpy(dev_a, h_m_elements, mbytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_b, h_n_elements, nbytes, cudaMemcpyHostToDevice);
    // // cudaMemcpy(dev_c, h_p_elements, pbytes, cudaMemcpyHostToDevice);

    // /* ------------------------------------
    //     Call Kernel Function
    //     ------------------------------------ */
    // // initial block and grid size
    // // int total_threads = mrow * ncolumn;
    // unsigned int grid_rows = (mrow + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // unsigned int grid_cols = (ncolumn + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // dim3 grid_size(grid_cols, grid_rows);
    // dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    // printf("Block size is: (%d, %d).\n", block_size.x, block_size.y);
    // printf("Grid size is: (%d, %d).\n", grid_size.x, grid_size.y);
    // // Executing kernel
    // matMulKernel <<< block_size, grid_size >>>(dev_a, dev_b, dev_c, mrow, mcolumn, ncolumn);
    // // Wait for GPU to finish before accessing on host
    // cudaDeviceSynchronize();
    // // Transfer data back to host memory
    // cudaMemcpy(h_p_elements, dev_c, pbytes, cudaMemcpyDeviceToHost);

    // // Demo matrix m,n,p
    // printf("\nMat p: \n");
    // for (int i = 0; i < mrow; i++)
    // {
    //     for(int j = 0; j < ncolumn; j++)
    //     {
    //         printf("%f ", h_p_elements[i * ncolumn + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\nMatMul Finished.\n");

    // // cudaFree(d_p);
    // free(h_p);
    return 0;
}

