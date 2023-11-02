// https://blog.csdn.net/zyboy2000/article/details/90338973
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
struct arg_struct{
    int *arg1;
    int *arg2;
}args;
pthread_t thread;
int conut = 0;
void *fn(void *arg)
{
    struct arg_struct *args;
    args = malloc(sizeof(struct arg_struct));
    args = arg;
    conut++;
    printf("start");
    printf("count %d, i = %d, j = %d\n",conut, *(args->arg1),*(args->arg2));
    free(args);
    return NULL;
}
int main()
{
    int err1;
    
    for (int u = '0'; u <= '9'; u++){
        for (int j = 6; j<7; j++){
            struct arg_struct *args = malloc(sizeof(struct arg_struct) * 1);
            int *p = (void *) malloc(sizeof(int));
            int *q = (void *) malloc(sizeof(int));
            *p = u;
            *q = j;
            args->arg1=p;
            args->arg2=q;
            err1 = pthread_create(&thread, NULL, fn, (void*)args);
        }
    }
    pthread_join(thread, NULL);
}
