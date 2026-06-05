#ifndef GEMM_COMMON_H
#define GEMM_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

typedef float DTYPE;

typedef struct
{
    double time_ms;
    double gflops;
} Result;

static inline double get_time()
{
    return (double)clock() / CLOCKS_PER_SEC;
}

#endif
