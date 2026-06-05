#include "gemm_functions.h"

Result gemm_openmp(DTYPE **A, DTYPE **B, DTYPE **C, int N)
{
    double start = get_time();

#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            DTYPE sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    double end = get_time();
    double time = (end - start) * 1000;
    double ops = 2.0 * N * N * N;
    double gflops = (ops / (time / 1000)) / 1e9;

    return (Result){time, gflops};
}
