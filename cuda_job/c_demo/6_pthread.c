// https://www.geeksforgeeks.org/multithreading-c-2/
// POSIX 表示可移植操作系统接口（Portable Operating System Interface ，缩写为 POSIX 是为了读音更像 UNIX）
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h> //Header file for sleep(). man 3 sleep for details.

// A normal C function that is executed as a thread 
// when its name is specified in pthread_create()
void *thread_function(void *arg)
{
    int i;
    for ( i = 0; i < 5; i++)
    {
        printf("Thread says hi!\n");
        sleep(1);
    }
    printf("Finish \n");
    return NULL;
}

int main(void)
{
    pthread_t thread_id;
    printf("Before Thread\n");
    // pthread_create( &thread_id, NULL, thread_function, NULL);
    if ( pthread_create( &thread_id, NULL, thread_function, NULL) )
    {
        printf("error creating thread.");
        abort();
    }
    // pthread_join ( thread_id, NULL );
    if ( pthread_join ( thread_id, NULL ) )
    {
        printf("error joining thread.");
        abort();
    }
    printf("After Thread\n");
    exit(0);
}
