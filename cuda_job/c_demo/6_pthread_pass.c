#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <malloc.h>

typedef void *(*ThreadFunc)(void *);

static volatile int running = 1;
static volatile char *enc;
static pthread_mutex_t mtx1;
static volatile int stepAdd = 0;
static volatile int stepSub = 0;

int count = 0;   // The number of combinations explored so far

void substr(char *dest, char *src, int start, int length)
{
    memcpy(dest, src + start, length);
    *(dest + length) = '\0';
}

void *crac(char *salt_and_encrypted)
{
    int x, y, z;     // Loop counters

    char salt[7];    // String used in hashing the password. Need space for \0
    char plain[4];   // The combination of letters currently being checked
    plain[4] = '\0'; // Put end of string marker on password
    char *enc;       // Pointer to the encrypted password

    substr(salt, salt_and_encrypted, 0, 6);
    char *_s = "ABC";

    while( running )
    {
        for(int x = 0; x <= 3; x++)
        {
            plain[0] = _s[x];
            for(int y = 0; y <= 3; y++)
            {
                plain[1] = _s[y];
                for(int z = 0; z <= 3; z++)
                {
                    plain[2] = _s[z];
                    for(int u = 0; u <= 3; u++)
                    {
                        plain[3] = _s[u];
                        enc = (char *) crypt(plain, salt);
                        count++;
                        if(strcmp(salt_and_encrypted, enc) == 0)
                        {
                            printf("#%-8d %s %s\n", count, plain, enc);
                            return enc;
                        }
                        else
                        {
                            printf(" %-8d %s %s\n", count, plain, enc);
                        }
                    }
                }
            }
        }
    }
    return NULL;
}


static int RunThread(pthread_t *handle, ThreadFunc f, void *context )
{
    char *result;
    pthread_attr_t thread_attribute;
    pthread_attr_init(&thread_attribute);
    pthread_attr_setschedpolicy(&thread_attribute, SCHED_RR);
    result = pthread_create(handle, &thread_attribute, f, context );
    pthread_attr_destroy(&thread_attribute);
    return result;
}

int main(int argc, const char *argv[])

{
    if(argc != 2)
    {
        printf("This program requires you to provide an encrypted password\n");
        return 1;
    }
    if(strlen(argv[1]) != 92)
    {
        printf("Encrypted passwords should be 92 characters long including salt\n");
        return 1;
    }
    char salt[7];    // String used in hashing the password. Need space for \0
    char plain[4];   // The combination of letters currently being checked
    plain[4] = '\0'; // Put end of string marker on password
    char *enc;       // Pointer to the encrypted password

    substr(salt, argv[1], 0, 6);
    crack(argv[1]);
    void *addResult = NULL;

    pthread_t addThread = 0;

    // Default mutex behavior sucks, btw.
    // It can auto-deadlock as it is not recursive.
    pthread_mutex_init(&mtx1, NULL);
    RunThread(&addThread, crac, NULL );

    while( running )
    {
        for(int x = 0; x <= 3; x++)
        {
            plain[0] = _s[x];
            plain[1] = _s[x];
            plain[2] = _s[x];
            plain[3] = _s[x];
        }
    }

    pthread_join( addThread, &addResult );


    puts( "All done - bye." );

    return 0;
}
// compiled with:
// clang++ -std=c++11 -stdlib=libc++ -lpthread baby_threading.cpp
// but should compile as C as well.