// A C program to show multiple threads with global and static variables

// As mentioned above, all threads share data segment. Global and static variables are stored in data segment. Therefore, they are shared by all threads. The following example program demonstrates the same.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

// Let us create a global variable to change it in threads
int g = 0;

char *_s = "ABC";
// The function to be executed by all threads
void *myThreadFun(int *vargp)
{
    // Store the value argument passed to this thread
    int *myid = (int *)vargp;
    char plain[4];   // The combination of letters currently being checked
    // Let us create a static variable to observe its changes
    plain[0] = _s[*myid];
    plain[1] = _s[*myid];
    plain[2] = _s[*myid];
    plain[3] = _s[*myid];
    plain[4] = '\0'; // Put end of string marker on password
    // Change static and global variables


    // Print the argument, static and global variables
    printf("Thread ID: %d, Static: %s, Global: %s\n", *myid, plain, plain);
    sleep(2);
    return NULL;
}

int main(int argc, char *argv[])
{
    pthread_t p[atoi(argv[1])];
    for (int i = 0; i < atoi(argv[1]); i++)
    {
        pthread_create(&p[i], NULL, &myThreadFun, (int *)i);
    }
    // plain[4] = '\0'; // Put end of string marker on password
    // Let us create three threads
    //     for (i = 0; i < atoi(argv[1]); i++)
    //         {
    //             plain[0] = _s[i];
    //             plain[1] = _s[i];
    //             plain[2] = _s[i];
    //             plain[3] = _s[i];
    // }
    pthread_exit(NULL);
    return 0;
}
// Please note that above is simple example to show how threads work. Accessing a global variable in a thread is generally a bad idea. What if thread 2 has priority over thread 1 and thread 1 needs to change the variable. In practice, if it is required to access global variable by multiple threads, then they should be accessed using a mutex.

