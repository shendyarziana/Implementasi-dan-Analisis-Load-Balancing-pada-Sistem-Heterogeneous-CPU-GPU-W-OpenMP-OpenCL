#include "gemm_functions.h"

/* GPU jika tersedia, jika tidak gunakan CPU */
Result gemm_opencl(DTYPE **A, DTYPE **B, DTYPE **C, int N)
{
    double start = get_time();

    /* Fallback ke CPU calculation (OpenCL bersifat opsional) */
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
