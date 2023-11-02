#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

float** load_matrix(size_t *_row_number, size_t *_column_number, const char *_file_path)
{

    *_row_number = 0;
    *_column_number = 0;

    /* open file */
    FILE *fp = fopen(_file_path, "r");
    if(fp == NULL)
    {
        fprintf(stderr, "Please check! No file %s, %s\n\n", _file_path, strerror(errno));
    }

    /* detect file content*/
    float **matrix = NULL, **tmp;
    char line[1024];

    /* read file */
    while(fgets(line, sizeof(line), fp))
    {
        /* determine columns size */
        if(*_column_number == 0)
        {
            char *scan = line;
            float dummy;
            int offset = 0;
            while(sscanf(scan, "%f%n", &dummy, &offset) == 1)
            {
                scan += offset;
                (*_column_number)++;
            }
        }

        tmp = realloc(matrix, (*_row_number + 1) * sizeof * matrix); // dynamic memory
        /* if no data */
        if(tmp == NULL)
        {
            fclose(fp);
        }
        matrix = tmp;
        matrix[*_row_number] = calloc(*_column_number, sizeof * matrix[*_row_number]); // allocate dynamic memory
        if(matrix[*_row_number] == NULL)
        {
            fclose(fp);
            if(*_row_number == 0)
            {
                fclose(fp);
            }
        }
        /* load data */
        int offset = 0;
        char *scan = line;
        for(size_t j = 0; j < *_column_number; j++)
        {
            if(sscanf(scan, "%f%n", matrix[*_row_number] + j, &offset) == 1)
                scan += offset;
            else
                matrix[*_row_number][j] = 0.0; // Missing then set 0
        }

        // incrementing _row_number
        (*_row_number)++;
    }
    fclose(fp);
    return matrix;
}

int main(int argc, char *argv[])
{
    printf("Load matrix \n");
    printf("Filename:%s\n", argv[1]);

    size_t _column_number, _row_number;
    float **matrix = load_matrix(&_row_number, &_column_number, argv[1]);

    for(size_t i = 0; i < _row_number; ++i)
    {
        for(size_t j = 0; j < _column_number; ++j)
            printf("%f ", matrix[i][j]);
        puts("");

    }
    printf("Row number %ld\n", _row_number);
    printf("Column number %ld\n", _column_number);

    return 0;
}