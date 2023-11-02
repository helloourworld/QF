#include "stdio.h" 


struct notify {
    void *data;
    char *name;
};
char get_name(char *sname)
{
    int ret;
    printf("3 sname addr = 0x%p\n", sname); //接受到传进来的参数后 将地址复制一份放到一个地方  
    *sname = "sxgd";
    return 0;
}
int main()
{
    int ret;
    struct notify noti;
    noti.data = NULL;
    noti.name = "abc";
    
    printf("1 name  = %s\n", noti.name);          //  传递参数之前 noti.name 的值
    printf("2 name addr = 0x%p\n", &noti.name);   //  传递之前noti.name的地址
    ret = get_name(&noti.name);     // 将noti.name的地址作为形参传送过去， 
    
    printf("ret = %d\n", ret);
    if(!ret){   
    printf("noti.name = %p\n",&noti.name);
    printf("noti.data = %s\n",noti.data);
    }
   return 0;
}

