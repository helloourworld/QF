/* ---------------------------------------------------
  My CUDA programming
  cd "/content/drive/Othercomputers/My MacBook Pro/cuda_job"
  --------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ char *CudaCrypt(char *toEncryptedWords)
{

    char *newPassword = (char *) malloc(sizeof(char) * 9);

    newPassword[0] = toEncryptedWords[0] - 2;
    newPassword[1] = toEncryptedWords[0] + 2;
    newPassword[2] = toEncryptedWords[1] + 1;
    newPassword[3] = toEncryptedWords[1] + 3;
    newPassword[4] = toEncryptedWords[2] - 3;
    newPassword[5] = toEncryptedWords[2] - 1;
    newPassword[6] = toEncryptedWords[3] + 2;
    newPassword[7] = toEncryptedWords[3] - 2;
    newPassword[8] = '\0';
    return newPassword;
}
/**
	Checks if one char string matches another
*/

__device__ int strCmp(const char *s1, const char *s2)
{
    while(*s1 && (*s1 == *s2))
    {
        s1++;
        s2++;
    }
    return *s1 - *s2;
}


/**
	Decrypts a pass using a CUDA thread
*/
__global__ void decryptPass(char *alphabet, char *numbers, char *encryptedWords, char *deviceOutputPass)
{
    /// Get the unique cuda thread id
    int uid = blockDim.x * blockIdx.x + threadIdx.x;

    /// Check if another thread found output pass before starting
    if(*deviceOutputPass != NULL)
    {
        printf("Found! stop other CUDA thread '%d'\n", uid);
        return;
    }

    /// Create potential pass to check on this thread
    char potentialPass[4];
    potentialPass[0] = alphabet[blockIdx.x];
    potentialPass[1] = alphabet[blockIdx.y];
    potentialPass[2] = numbers[threadIdx.x];
    potentialPass[3] = numbers[threadIdx.y];

    /// Encrypt the potential password
    char *encryptedPotential;
    encryptedPotential = CudaCrypt(potentialPass);

    // printf("UID: '%d' Plain: '%s' Encrypted Plain: '%s' Target Encrypted: '%s'\n", uid, potentialPass, encryptedPotential, encryptedWords);

    /// Check the current potential pass is matches the target encryptedWords
    if ( strCmp(encryptedWords, encryptedPotential) == 0 )
    {
        printf("UID '%d', Encrypted pass '%s', '%s' matches potential pass = '%s'\n", uid, encryptedWords, potentialPass, encryptedPotential);
        for (int i = 0; i < 4; i++ )
        {
            deviceOutputPass[i] = potentialPass[i];
        }
    }
}

/**

*/
int main(int argc, char *argv[])
{

    /// Get the encrypted pass to decrypt
    char *encryptedWords;
    if (argc > 1)
    {
        encryptedWords = argv[1];
    }
    else{
    	encryptedWords = "fjjl/12.";
    }

    printf("The EncryptedPasswords: '%s'\n", encryptedWords);

    // Init alphabet and numbers array to read only use in cuda
    char cpuAlphabet[26*2] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' ,
     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };
    
    char cpuNumbers[10] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };

    int sizeOfEncryptEdpass = sizeof(char) * 9;

    char *gpuAlphabet;
    cudaMalloc( (void **) &gpuAlphabet, sizeof(char) * 26 * 2 );
    cudaMemcpy( gpuAlphabet, cpuAlphabet, sizeof(char) * 26 * 2, cudaMemcpyHostToDevice );

    char *gpuNumbers;
    cudaMalloc( (void **) &gpuNumbers, sizeof(char) * 10 );
    cudaMemcpy( gpuNumbers, cpuNumbers, sizeof(char) * 10, cudaMemcpyHostToDevice );

    char *gpuEncryptEdpass;
    cudaMalloc( (void **) &gpuEncryptEdpass, sizeOfEncryptEdpass );
    cudaMemcpy( gpuEncryptEdpass, encryptedWords, sizeOfEncryptEdpass, cudaMemcpyHostToDevice);

    char *gpuOutputPass;
    cudaMalloc( (void **) &gpuOutputPass, sizeOfEncryptEdpass );


    /// Launch cuda threads and await finish
    decryptPass <<< dim3(26 * 2, 26 * 2, 1), dim3(10, 10, 1) >>>(gpuAlphabet, gpuNumbers, gpuEncryptEdpass, gpuOutputPass);
    cudaDeviceSynchronize();

    /// Copy GPU output pass to the CPU
    char *cpuOutputPass = (char *)malloc( sizeof(char) * 4 );
    cudaMemcpy(cpuOutputPass, gpuOutputPass, sizeOfEncryptEdpass, cudaMemcpyDeviceToHost);

    /// If output pass contained an output, print the results
    printf("\nCrack Results:\n");
    if (cpuOutputPass != NULL && cpuOutputPass[0] != 0) {
        printf("Given Encrypted Pass: '%s'\n", encryptedWords);
        printf("Found Decrypted Pass: '%s'\n", cpuOutputPass);
    }
    else
    {
        printf("Unable to find a password.\n");
    }

    /// Free any malloc'd memory
    cudaFree(gpuAlphabet);
    cudaFree(gpuNumbers);
    cudaFree(gpuEncryptEdpass);
    cudaFree(gpuOutputPass);
    free(cpuOutputPass);
}

