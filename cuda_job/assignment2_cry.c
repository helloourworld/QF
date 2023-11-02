#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <malloc.h>
#include <pthread.h>

/******************************************************************************
  Compile with:
    cc -o CrackL4 CrackL4.c -lcrypt
******************************************************************************/

/**
 Required by lack of standard function in C.
*/

void substr(char *dest, char *src, int start, int length)
{
    memcpy(dest, src + start, length);
    *(dest + length) = '\0';
}

int points_per_thread = -1;
int total_hits = 0;


void *compute_pi(void *arg)
{
    int thread_id = (int) arg;
    unsigned int rstate = 123456789 * thread_id;
    int local_hits = 0;
    for (int i = 0; i < points_per_thread; i++)
    {
        double x = ((double) rand_r(&rstate)) / ((double) RAND_MAX);
        double y = ((double) rand_r(&rstate)) / ((double) RAND_MAX);
        if (x * x + y * y <= 1.0)
        {
            local_hits++;
        }
    }
    return (void *) local_hits;
}


/**
 This function can crack a three letter lowercase password. All combinations
 that are tried are displayed and when the password is found, #, is put at the
 start of the line. Note that one of the most time consuming operations that
 it performs is the output of intermediate results, so performance experiments
 for this kind of  program should not include this.
*/

void crack(char *salt_and_encrypted)
{
    int x, y, z;     // Loop counters
    int count = 0;   // The number of combinations explored so far
    char salt[7];    // String used in hashing the password. Need space for \0
    char plain[4];   // The combination of letters currently being checked
    plain[4] = '\0'; // Put end of string marker on password
    char *enc;       // Pointer to the encrypted password

    substr(salt, salt_and_encrypted, 0, 6);
    char *_s = "ABC";
    for(int x = 0; x <= 3; x++)
    {
        plain[0] = _s[x];
        for(int y = 0; y <= 3; y++)
        {
            plain[1] = _s[y];
            for(int z = 0; z <= 9; z++)
            {
                plain[2] = z;
                for(int u = 0; u <= 9; u++)
                {
                    plain[3] = u;

                    enc = (char *) crypt(plain, salt);
                    count++;
                    if(strcmp(salt_and_encrypted, enc) == 0)
                    {
                        printf("#%-8d %s %s\n", count, plain, enc);
                        exit(0);
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

int main(int argc, char *argv[])
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

    crack(argv[1]);

    return 0;
}