#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#define N 10

void vector_add(float *out, float *a, float *b, int n)
{
    for(int i = 0; i < n; i++)
    {
        out[i] = a[i] + b[i];
    }
}

int main()
{
    float *a, *b, *out;

    // Allocate memory
    a   = (void *)malloc(sizeof(float) * N);
    b   = (void *)malloc(sizeof(float) * N);
    out = (void *)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = pow(i, 2);
    }

    // Main function
    vector_add(out, a, b, N);

    printf("%f %f\n", out[0], out[N - 1]);
}
