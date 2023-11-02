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
#include <string.h>
#include <errno.h>
#include <cstdlib>

#define N 100
#define MAXERR 1e-6
// Thread block size
#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

/* ------------------------------------
   1 Read data from file appropriately
   ------------------------------------ */

void load_matrix(size_t *_row_number, size_t *_column_number, const char *_file_path)
{
    /* parameters check */
    if(_row_number == NULL || _column_number == NULL || _file_path == NULL)
        return 1;

    *_row_number = 0;
    *_column_number = 0;

    /* open file */
    FILE *fp = fopen(_file_path, "r");
    if(fp == NULL)
    {
        fprintf(stderr, "Please check! No file %s, %s\n\n", _file_path, strerror(errno));
        return 1;
    }

    /* detect file content*/
    float **matrix = NULL, **tmp;
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

        tmp = realloc(matrix, (*_row_number + 1) * sizeof * matrix); // dynamic memory
        /* if no data */
        if(tmp == NULL)
        {
            fclose(fp);
            return matrix;
        }
        matrix = tmp;
        matrix[*_row_number] = calloc(*_column_number, sizeof * matrix[*_row_number]); // allocate dynamic memory
        if(matrix[*_row_number] == NULL)
        {
            fclose(fp);
            if(*_row_number == 0)
            {
                fclose(fp);
                free(matrix);
                return NULL;
            }
            return matrix;
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
    }
    setElement(C, row, col, Cvalue);
}


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
int main(int argc, char *argv[])
{
    // Load data
    printf("Load matrix \n");
    printf("Filename:%s\n", argv[1]);

    size_t _column_number, _row_number;
    float **matrix = load_matrix(&_row_number, &_column_number, argv[1]);

    if(matrix == NULL)
    {
        fprintf(stderr, "Please check! No matrix.\n\n");
        return 1;
    }


    for(size_t i = 0; i < _row_number; ++i)
    {
        for(size_t j = 0; j < _column_number; ++j)
            printf("%f ", matrix[i][j]);
        puts("");

    }
    printf("Row number %ld\n", _row_number);
    printf("Column number %ld\n", _column_number);
    
    /* ------------------------------------
       Call Kernel Function
       ------------------------------------ */
    // initial block and grid size

    // dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 gridSize(B->width / blockSize.x, A->height / blockSize.y);

    /* ------------------------------------
    Call the hello( ) kernel function
    ------------------------------------ */
    // matMulKernel << < gridSize, blockSize >> >(A, B, C);


    // Transfer data back to host memory
    // cudaMemcpy(C->elements, d_C->elements, nBytes,
    //            cudaMemcpyDeviceToHost);
    

    // printf("Hello World ! \n");

    // printf("Matrix[2] = %d\n", Matrix[2]);
    printf("PASS");

    // Deallocate device memory
    // cudaFree(Matrix);

    // Deallocate host memory
    // free(A);
    // return 0;
}
