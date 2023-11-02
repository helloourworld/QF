#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// confused
int main()
{
    char *str;
    /* Inintial memory allocate*/
    str = (char *) malloc(15);
    strcpy(str, "runoob");
    printf("String = %s, Address = %u %u\n", str, str, &str);

    /* reallocate*/
    str = (char *) realloc(str, 25);
    strcat(str, ".com");
    printf("String = %s, Address = %u %u\n", str, str, &str);
    free(str);
    return 0;
}
