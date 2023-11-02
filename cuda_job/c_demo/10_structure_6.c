#include <stdio.h>
#include <stdlib.h>
typedef struct person
{
    int age;
    float weight;
    char name[3];
} P;

int main()
{
    struct person *ptr;
    int i, n;

GO:
    printf("Enter the number of persons: ");
    scanf("%d", &n);

    if (n > 5)
    {
        printf("too large");
        goto GO;
    }

    // allocating memory for n numbers of struct person
    ptr = (struct person *) malloc(n * sizeof(struct person));

    for(i = 0; i < n; ++i)
    {
        printf("Enter first name and age respectively: ");

        // To access members of 1st struct person,
        // ptr->name and ptr->age is used

        // To access members of 2nd struct person,
        // (ptr+1)->name and (ptr+1)->age is used
        scanf("%s %d", (ptr + i)->name, &(ptr + i)->age);
    }

    printf("Displaying Information:\n");
    for(i = 0; i < n; ++i)
        printf("Name: %s\tAge: %d\n", (ptr + i)->name, (ptr + i)->age);

    return 0;
}

// In the above example, n number of struct variables are created where n is entered by the user.

// To allocate the memory for n number of struct person, we used,

// ptr = (struct person*) malloc(n * sizeof(struct person));
// Then, we used the ptr pointer to access elements of person.