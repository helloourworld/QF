// https://stackoverflow.com/questions/33658228/c-print-array
#include <stdio.h>

int main (void) {

    int numbers[100] = { 0 };
    int c = 0;
    int i = 0;
    char answer[30] = { 0 };

    printf (" Please insert a value: ");

GO:

    if (scanf ("%d", &numbers[c]) == 1)
        c++;
    getchar ();
    printf (" Do you want to add another value (y/n)? ");
    scanf ("%c", answer);
    if (*answer == 'y') {
        printf (" Please insert another value: ");
        goto GO;
    }

    for (i = 0; i < 40; i++) {
        printf (" number[%2d] : %d\n", i, answer[i]);
    }

    return 0;
}