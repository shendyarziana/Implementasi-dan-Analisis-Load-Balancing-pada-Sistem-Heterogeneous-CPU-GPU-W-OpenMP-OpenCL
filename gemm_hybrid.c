#include "gemm_functions.h"

Result gemm_hybrid(DTYPE **A, DTYPE **B, DTYPE **C, int N, float cpu_ratio)
{
    double start = get_time();

    int cpu_rows = (int)(N * cpu_ratio);
    int gpu_rows = N - cpu_rows;

/* CPU part */
#pragma omp parallel for
    for (int i = 0; i < cpu_rows; i++)
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

    /* GPU part (fallback ke CPU jika GPU tidak tersedia) */
    for (int i = cpu_rows; i < N; i++)
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
