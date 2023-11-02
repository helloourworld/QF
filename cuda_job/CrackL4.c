#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <malloc.h>

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
    char *_s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for(int x = 0; x <= 26 * 2 + 10 - 1; x++)
    {
        plain[0] = _s[x];
        for(int y = 0; y <= 26 * 2 + 10 - 1; y++)
        {
            plain[1] = _s[y];
            for(int z = 0; z <= 26 * 2 + 10 - 1; z++)
            {
                plain[2] = _s[z];
                for(int u = 0; u <= 26 * 2 + 10 - 1; u++)
                {
                    plain[3] = _s[u];

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