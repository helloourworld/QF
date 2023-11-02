#include<stdio.h>

int main()
{
    int k;
    int *ip;
    ip = &k;  //给ip赋值 store k's address into point var ip
    *ip = 7; //给ip所指向的内存赋值，即 k= 7
    printf("Address of variable: %p\n", &k);
    printf("Address stored in ip variable: %p\n", ip);
    printf("Address stored for ip variable: %p\n", &ip);
    /* visit value through point*/
    printf("Value of *ip variable: %x\n", *ip);
    printf("Value of *ip variable: %x\n", k);

    // NULL pointer
    int *ptr = NULL;
    printf("ptr value is %p (%u)\n", ptr, ptr);
    return 0;
}
