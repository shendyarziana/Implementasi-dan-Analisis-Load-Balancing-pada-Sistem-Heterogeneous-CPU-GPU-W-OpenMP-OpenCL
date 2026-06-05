#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gemm_functions.h"

int main(int argc, char *argv[])
{
    int N = 512;
    if (argc > 1)
        N = atoi(argv[1]);

    printf("GEMM Benchmark - Matrix Size: %d x %d\n", N, N);
    printf("========================================\n\n");

    /* Allocate matrices */
    DTYPE **A, **B, **C;
    allocate_matrix(&A, N);
    allocate_matrix(&B, N);
    allocate_matrix(&C, N);

    /* Initialize */
    srand(42);
    init_matrix(A, N);
    init_matrix(B, N);

    /* Run benchmarks */
    printf("Method              Time (ms)    GFLOPs    Speedup\n");
    printf("------------------------------------------------------\n");

    Result serial = gemm_serial(A, B, C, N);
    printf("Serial            %10.3f   %8.2f    1.00x\n", serial.time_ms, serial.gflops);

    Result openmp = gemm_openmp(A, B, C, N);
    printf("OpenMP            %10.3f   %8.2f   %.2fx\n", openmp.time_ms, openmp.gflops, serial.time_ms / openmp.time_ms);

    Result opencl = gemm_opencl(A, B, C, N);
    printf("OpenCL (CPU)      %10.3f   %8.2f   %.2fx\n", opencl.time_ms, opencl.gflops, serial.time_ms / opencl.time_ms);

    Result hybrid = gemm_hybrid(A, B, C, N, 0.5);
    printf("Hybrid (50-50)    %10.3f   %8.2f   %.2fx\n", hybrid.time_ms, hybrid.gflops, serial.time_ms / hybrid.time_ms);

    printf("\n");

    /* Cleanup */
    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
