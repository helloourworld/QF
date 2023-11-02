#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef void *(*ThreadFunc)(void *);

static volatile int running = 1;
static volatile int result = 0;
static pthread_mutex_t mtx1;
static volatile int stepAdd = 0;
static volatile int stepSub = 0;

void *add( void *num)
{
    int random;

    while( running )
    {
        if( result < 101 )
        {
            pthread_mutex_lock(&mtx1);
            random = rand() % 51;
            result += random;
            stepAdd += 1;
            pthread_mutex_unlock(&mtx1);
            //printf("%d. [random -> %d, result now -> %d] ADD\n",step,random,result);
        }
        // else - one could avoid some useless spinning of the thread. (pthread_yield())
    }
    pthread_exit(NULL);
}

void *sub( void *num)
{
    int random;
    while( running )
    {
        if( result > 5 )
        {
            pthread_mutex_lock(&mtx1);
            random = rand() % 51;
            result -= random;
            stepSub += 1;
            pthread_mutex_unlock(&mtx1);
            //          printf("%d. [random -> %d, result now -> %d] ADD\n",step,random,result);
        }
    }
    pthread_exit(NULL);
}



static int RunThread(pthread_t *handle, ThreadFunc f, void *context )
{
    int result;
    pthread_attr_t thread_attribute;
    pthread_attr_init(&thread_attribute);
    pthread_attr_setschedpolicy(&thread_attribute, SCHED_RR);
    result = pthread_create(handle, &thread_attribute, f, context );
    pthread_attr_destroy(&thread_attribute);
    return result;
}

int main(int argc, const char *argv[])
{
    void *addResult = NULL;
    void *subResult = NULL;
    pthread_t addThread = 0;
    pthread_t subThread = 0;
    // Default mutex behavior sucks, btw.
    // It can auto-deadlock as it is not recursive.
    pthread_mutex_init(&mtx1, NULL);
    RunThread(&addThread, add, NULL );
    RunThread(&subThread, sub, NULL );

    while( running )
    {
        if( 13 == result )
        {
            running = 0;
        }
        else
        {
            printf("Steps: add(%d) sub(%d) -- result = %d\n", stepAdd, stepSub, result );
        }
    }

    pthread_join( addThread, &addResult );
    pthread_join( subThread, &subResult );

    puts( "All done - bye." );

    return 0;
}
// compiled with:
// clang++ -std=c++11 -stdlib=libc++ -lpthread baby_threading.cpp
// but should compile as C as well.