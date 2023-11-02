#include <stdio.h>

void h(int n)
{
    float a[10];
    for (int i = 0; i < n; i++)
    {
        a[i] = i * 10;
        // printf("%f\n", a[i]);
    }
    char * str_possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for(int x = 0; x <26*2+10; x++){
        printf("%d, %c\n", str_possible[x], str_possible[x]);
    }
}

int main()
{
    h(5);
    return 0;
}
