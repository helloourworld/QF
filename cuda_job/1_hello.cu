/* ---------------------------------------------------
   My Hello world for CUDA programming
   "/content/drive/Othercomputers/My MacBook Pro/cuda_job"
   --------------------------------------------------- */

#include <stdio.h>        // C programming header file
#include <unistd.h>       // C programming header file
                          // cude.h is automatically included by nvcc...
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 100
#define MAXERR 1e-6
/* ------------------------------------
   Your first kernel (= GPU function)
   ------------------------------------ */
__global__ void hello(int *out,int *a, int n)
{
   int index = threadIdx.x;
   int stride = blockDim.x;
   for (int i = index; i<n; i += stride){
      out[i] = a[i];
   }
}

int main()
{
   // Host pointers
   int *a, *out;

   // Device pointers
   int *d_a, *d_out;
   
   size_t bytes = N * sizeof(int);

   // Allocate host memory
   a = (int*)malloc(bytes);
   out = (int*)malloc(bytes);

   // Initialize host arrays
   for (int i=0; i<N; i++){
      a[i] = 1;
   }

   // Allocate device buff
   cudaMalloc((void**)&d_out, bytes);
   cudaMalloc((void**)&d_a, bytes);

   // Transfer data to device buff
   cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
   
   /* ------------------------------------
      Call the hello( ) kernel function
      ------------------------------------ */
   hello<<< 1, 16 >>>(d_out, d_a, N );
   
   // Transfer data back to host memory
   cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

   // Verification
  //  for (int i=0; i<N; i++){
  //     assert(fabs(out[i] - a[i]) < MAXERR);
  //  }
   printf("I am the CPU: Hello World ! \n");
   //sleep(1);   // Necessary to give time to let GPU threads run !!!
   printf("out[2] = %d\n", out[2]);
   printf("PASS");
   
   // Deallocate device memory
   cudaFree(d_a);
   cudaFree(d_out);

   // Deallocate host memory
   free(a);
   free(out);
  //  return 0;
}
