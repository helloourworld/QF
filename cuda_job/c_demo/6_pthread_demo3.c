# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <pthread.h>

#define BIG 10000000UL

int counter = 0;

void *workerThreadFunc(void *var)
{

    for (int i = 0; i < BIG; i++)
    {
        counter++;
    }
    return NULL;
}


int main()
{

    pthread_t tid0;
	pthread_create(&tid0, NULL, workerThreadFunc, (void *)&tid0);
	workerThreadFunc(NULL);
    pthread_join(tid0, NULL);
    printf("Done. counter = %u\n", counter);
    // pthread_exit(NULL);
    return 0;
}