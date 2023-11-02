
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *CudaCrypt(char *rawPassword)
{

    char *newPassword = (char *) malloc(sizeof(char) * 9);

    newPassword[0] = rawPassword[0] - 2;
    newPassword[1] = rawPassword[0] + 2;
    newPassword[2] = rawPassword[1] + 1;
    newPassword[3] = rawPassword[1] + 3;
    newPassword[4] = rawPassword[2] - 3;
    newPassword[5] = rawPassword[2] - 1;
    newPassword[6] = rawPassword[3] + 2;
    newPassword[7] = rawPassword[3] - 2;
    newPassword[8] = '\0';

    // for(int i =0; i<10; i++){
    // 	if(i >= 0 && i < 6){ //checking all lower case letter limits
    // 		if(newPassword[i] > 122){
    // 			newPassword[i] = (newPassword[i] - 122) + 97;
    // 		}else if(newPassword[i] < 97){
    // 			newPassword[i] = (97 - newPassword[i]) + 97;
    // 		}
    // 	}else{ //checking number section
    // 		if(newPassword[i] > 57){
    // 			newPassword[i] = (newPassword[i] - 57) + 48;
    // 		}else if(newPassword[i] < 48){
    // 			newPassword[i] = (48 - newPassword[i]) + 48;
    // 		}
    // 	}
    // }
    return newPassword;
}


int main()
{
    // /// Get the unique cuda thread id
    // int uid = blockDim.x * blockIdx.x + threadIdx.x;

    // /// Check if another thread found output pass before starting
    // if(*deviceOutputPass != NULL) {
    // 	//printf("OutputPass not null! Cancelling CUDA thread '%d'\n", uid);
    // 	return;
    // }

    // /// Create potential pass to check on this thread
    // char potentialPass[4];
    // potentialPass[0] = alphabet[blockIdx.x];
    // potentialPass[1] = alphabet[blockIdx.y];
    // potentialPass[2] = numbers[threadIdx.x];
    // potentialPass[3] = numbers[threadIdx.y];
    char *potentialPass = "hi20";
    // /// Encrypt the potential password
    char *encryptedPotential;
    encryptedPotential = CudaCrypt(potentialPass);

    printf("UID: '%d' Plain: '%s' Encrypted Plain: '%s' Target Encrypted: '%s'\n", "uid", potentialPass, encryptedPotential, "encryptedPass");

    if(strcmp(encryptedPotential, "\?CBD68;7") == 0)
        {
            printf("done");
        }
    /// Check the current potential pass is matches the target encryptedPass
    // if ( isEncryptedMatching(encryptedPass, encryptedPotential, 11) > 0 )
    // {
    // 	/// Matches so set deviceOutputPass to the current combination
    // 	printf("UID '%d' Encrypted pass '%s' from combination '%s' matches potential pass = '%s'\n", uid, encryptedPass, potentialPass, encryptedPotential);
    // 	for (int i = 0; i < 4; i++ ) {
    // 		deviceOutputPass[i] = potentialPass[i];
    // 	}
    // }
}