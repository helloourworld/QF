#include<stdio.h>
int main(void) 
{ 

    char a[0];
    int b;
    printf("请输入三个字符：");
    scanf("%s %d",a, &b); 
    printf("%s, %d\n", a, b);
    return 0;
}

// 首先说明 %s格式符 表示用来输入出一个字符串 而字符串是以数组的形式的存储的
// c语言中数组名代表该数组的起始地址。 此处,a为数组名 代表的是首地址，所以就不用取地址符了， 再用取地址符号 就重复了 请注意与**%c**的区别 理解就好啦。

// 当然带上也不为错。