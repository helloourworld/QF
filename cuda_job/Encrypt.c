#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <crypt.h>

/****************************************************************************** 
  Compile with:
    cc -o EncryptSHA512 Encrypt.c -lcrypt
    
  To encrypt the password "pass":
    ./EncryptSHA512 pass
******************************************************************************/

#define SALT "$6$KB$"

int main(int argc, char *argv[]){
  
  printf("%s\n", crypt(argv[1], SALT));

  return 0;
}