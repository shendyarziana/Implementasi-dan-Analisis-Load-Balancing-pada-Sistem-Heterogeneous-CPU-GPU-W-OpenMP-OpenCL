#include "gemm_common.h"

void allocate_matrix(DTYPE ***matrix, int size)
{
    *matrix = (DTYPE **)malloc(size * sizeof(DTYPE *));
    for (int i = 0; i < size; i++)
    {
        (*matrix)[i] = (DTYPE *)malloc(size * sizeof(DTYPE));
    }
}

void free_matrix(DTYPE **matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

void init_matrix(DTYPE **matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = (DTYPE)(rand() % 100) / 10.0f;
        }
    }
}
