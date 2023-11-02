#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <malloc.h>
#include <pthread.h>
#include <unistd.h>
/******************************************************************************
  Compile with:
    gcc .c -lcrypt -pthread
    ./a.out \$6\$KB\$yQGJw.Sdx7s6aoXdtCWYUPmABAulKV8kR11LNS7sDqdXm49UzEJgoF59WuYjM5zTWEcc/YA208QToKjJdxFbL/ >results.txt
******************************************************************************/

// AA99  $6$KB$QJBvuJ32BlT7myD.0hFzokn6xk92zwoQR2XVZqM0NcO/bcFxZ/5dpMOYX24Ujp92hbGYWCLtSXeXqM3ZSHQR..
static volatile int running = 1;
static volatile int result = 0;
struct arg_struct
{
    char *arg1;
    int *plain1;
    int *plain2;
    int *plain3;
    int *plain4;
} args;

void substr(char *dest, char *src, int start, int length)
{
    memcpy(dest, src + start, length);
    *(dest + length) = '\0';
}

void *crack(void *arguments)
{
    static int count = 0;   // The number of combinations explored so far

    char *enc;       // Pointer to the encrypted password
    char salt[7];    // String used in hashing the password. Need space for \0
    struct arg_struct *args;
    args = malloc(sizeof(struct arg_struct) * 1);
    args = arguments;
    substr(salt, args->arg1, 0, 6);

    char *_s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    while (running)
    {
        char plain[5];   // The combination of letters currently being checked
        plain[4] = '\0'; // Put end of string marker on password
        plain[0] = _s[*(args->plain1)];
        plain[1] = _s[*(args->plain2)];
        plain[2] = *(args->plain3);
        plain[3] = *(args->plain4);
        char *pw = (char *) malloc(sizeof(char) * (strlen(plain) + 1));
        strcpy(pw, plain);
        enc = (char *) crypt(plain, salt);
        char *encode = (char *) malloc(sizeof(char) * (strlen(enc) + 1));
        strcpy(encode, enc);
        count++;
        pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
        if(strcmp(args->arg1, encode) == 0)
        {
            result = 1;
            printf("#%-8d %s  %s %d\n", count, pw, encode, result);
            printf("thread set for cancel\n");
            pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
        }
        else
        {
            printf("go...%d\n", result);
        }
        // free(args);
        free(pw); // do not forget to free allocated memory    free(encode);
        free(encode); // do not forget to free allocated memory    free(encode);
        // free(args);
        sleep(0.1);
        pthread_exit(NULL);
    }

    return NULL;
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


    pthread_t some_thread;


    for(int x = 0; x <= 2 * 26; x++)
    {
        int *p = (void *) malloc(sizeof(int));
        *p = x;
        // plain[0] = _s[x];
        for(int y = 0; y <= 2 * 26; y++)
        {
            int *q = (void *) malloc(sizeof(int));
            *q = y;
            for(int z = '0'; z <= '9'; z++)
            {
                int *r = (void *) malloc(sizeof(int));
                *r = z;
                for(int u = '0'; u <= '9'; u++)
                {
                    if( 1 == result )
                    {
                        running = 0;
                    }
                    else
                    {
                        int *s = (void *) malloc(sizeof(int));
                        *s = u;
                        struct arg_struct *args = malloc(sizeof(struct arg_struct) * 1);
                        args->arg1 = argv[1];
                        args->plain1 = p;
                        args->plain2 = q;
                        args->plain3 = r;
                        args->plain4 = s;
                        pthread_create(&some_thread, NULL, &crack, (void *)args);
                    }
                }
            }
        }
    }
    pthread_cancel(some_thread);
    pthread_join(some_thread, NULL);


    return 0; /* Wait until thread is finished */
}