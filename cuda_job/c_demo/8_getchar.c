#include <stdio.h>

int main ()
{
   char c;
 
   printf("请输入字符：");
   c = getchar();
 
   printf("输入的字符：");
   putchar(c);

   return(0);
}