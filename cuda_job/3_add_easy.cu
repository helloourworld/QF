#include "stdio.h"
__global__ void add(int a, int b, int *c)
{
*c = a + b; }

int main() {
int a,b,c;
int *dev_c;
int *dev_a;
int *dev_b;
a=3;
b=4;

size_t bytes = sizeof(int);
cudaMalloc((void**)&dev_a, bytes);
cudaMalloc((void**)&dev_b, bytes);
cudaMalloc((void**)&dev_c, sizeof(int)); 
   // Transfer data to device buff
//cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);
//cudaMemcpy(dev_b, &b, sizeof(int), cudaMemcpyHostToDevice);

add<<<1,1>>>(a,b,dev_c);
cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost); 
printf("%d + %d is %d\n", a, b, c);
cudaFree(dev_c);
return 0;
}