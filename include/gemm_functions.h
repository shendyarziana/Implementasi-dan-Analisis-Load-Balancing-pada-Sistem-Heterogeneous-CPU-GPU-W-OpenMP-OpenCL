#ifndef GEMM_FUNCTIONS_H
#define GEMM_FUNCTIONS_H

#include "gemm_common.h"

/* GEMM Implementations */
Result gemm_serial(DTYPE **A, DTYPE **B, DTYPE **C, int N);
Result gemm_openmp(DTYPE **A, DTYPE **B, DTYPE **C, int N);
Result gemm_opencl(DTYPE **A, DTYPE **B, DTYPE **C, int N);
Result gemm_hybrid(DTYPE **A, DTYPE **B, DTYPE **C, int N, float cpu_ratio);

/* Utility Functions */
void allocate_matrix(DTYPE ***matrix, int size);
void free_matrix(DTYPE **matrix, int size);
void init_matrix(DTYPE **matrix, int size);

#endif /* GEMM_FUNCTIONS_H */
