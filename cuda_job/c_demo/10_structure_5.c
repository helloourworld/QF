// https://www.programiz.com/c-programming/c-structures-pointers

#include <stdio.h>
struct person
{
   int age;
   float weight;
};

int main()
{
    struct person *personPtr, person1;
    personPtr = &person1;   

    printf("Enter age: ");
    scanf("%d", &personPtr->age);

    printf("Enter weight: ");
    scanf("%f", &personPtr->weight);

    printf("Displaying:\n");
    printf("Age: %d\n", (*personPtr).age);
    printf("weight: %f\n", (*personPtr).weight);

    return 0;
}

// In this example, the address of person1 is stored in the personPtr pointer using personPtr = &person1;.

// Now, you can access the members of person1 using the personPtr pointer.

// By the way,

// personPtr->age is equivalent to (*personPtr).age
// personPtr->weight is equivalent to (*personPtr).weight