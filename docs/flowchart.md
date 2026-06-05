# 🔄 Program Flowchart

Dokumentasi visual untuk alur eksekusi GEMM benchmark.

## Main Program Execution Flow

```
START: main()
│
├─ Parse Arguments
│  ├─ Input: matrix size N (256, 512, or 1024)
│  └─ Validate: N > 0 and reasonable memory
│
├─ Memory Allocation
│  ├─ A = malloc(N² × sizeof(float))
│  ├─ B = malloc(N² × sizeof(float))
│  └─ C = malloc(N² × sizeof(float))
│
├─ Initialize Matrices
│  ├─ Fill A with random values (0.0 - 1.0)
│  ├─ Fill B with random values (0.0 - 1.0)
│  └─ Set C = 0
│
├─ BENCHMARK LOOP (4 methods)
│  │
│  ├─ [1] Serial GEMM
│  │   ├─ Start timer
│  │   ├─ Call gemm_serial(A, B, C, N)
│  │   │   └─ Triple nested loop (i,j,k)
│  │   │       └─ C[i,j] += A[i,k] × B[k,j]
│  │   ├─ Stop timer → Record time_serial
│  │   └─ Verify correctness (reference)
│  │
│  ├─ [2] OpenMP GEMM
│  │   ├─ Start timer
│  │   ├─ Call gemm_openmp(A, B, C, N)
│  │   │   ├─ #pragma omp parallel for collapse(2)
│  │   │   ├─ Distribute (i,j) pairs to threads
│  │   │   ├─ Each thread:
│  │   │   │   └─ Inner loop k, compute C[i,j] += A[i,k] × B[k,j]
│  │   │   └─ Implicit barrier at end
│  │   ├─ Stop timer → Record time_openmp
│  │   └─ Verify correctness vs serial
│  │
│  ├─ [3] OpenCL GEMM
│  │   ├─ Start timer
│  │   ├─ Call gemm_opencl(A, B, C, N)
│  │   │   ├─ Get OpenCL device (GPU or CPU fallback)
│  │   │   ├─ Create command queue
│  │   │   ├─ Create memory buffers:
│  │   │   │   ├─ d_A = clCreateBuffer (copy A)
│  │   │   │   ├─ d_B = clCreateBuffer (copy B)
│  │   │   │   └─ d_C = clCreateBuffer (empty)
│  │   │   ├─ Compile kernel from source
│  │   │   ├─ Enqueue kernel:
│  │   │   │   ├─ Global work size: (N, N)
│  │   │   │   ├─ Each work-item (i,j) computes:
│  │   │   │   │   └─ C[i,j] += A[i,k] × B[k,j]
│  │   │   │   └─ clEnqueueNDRangeKernel()
│  │   │   ├─ Read back result: clEnqueueReadBuffer(d_C → C)
│  │   │   ├─ clFinish() - wait for completion
│  │   │   └─ Release GPU resources
│  │   ├─ Stop timer → Record time_opencl
│  │   └─ Verify correctness vs serial
│  │
│  └─ [4] Hybrid GEMM
│      ├─ Start timer
│      ├─ Call gemm_hybrid(A, B, C, N, cpu_ratio)
│      │   ├─ Calculate split:
│      │   │   ├─ cpu_rows = (int)(N × HYBRID_CPU_RATIO)
│      │   │   └─ gpu_rows = N - cpu_rows
│      │   │
│      │   ├─ Thread 1: CPU computation (OpenMP)
│      │   │   └─ Compute C[0..cpu_rows-1, 0..N-1]
│      │   │
│      │   ├─ Thread 2: GPU computation (OpenCL)
│      │   │   └─ Compute C[cpu_rows..N-1, 0..N-1]
│      │   │
│      │   └─ Synchronize (wait both complete)
│      │
│      ├─ Stop timer → Record time_hybrid
│      └─ Verify correctness vs serial
│
├─ Calculate Performance Metrics
│  └─ For each method:
│     ├─ GFLOPs = (2×N³) / (10⁹ × time_ms / 1000)
│     ├─ Speedup = time_serial / time_method
│     └─ Efficiency = Speedup / NUM_CORES
│
├─ Print Results Table
│  ├─ Method | Time (ms) | GFLOPs | Speedup | Efficiency
│  ├─ Serial |    ...    |  ...   |  1.00x  |   100%
│  ├─ OpenMP |    ...    |  ...   |  ...x   |    ...%
│  ├─ OpenCL |    ...    |  ...   |  ...x   |    ...%
│  └─ Hybrid |    ...    |  ...   |  ...x   |    ...%
│
├─ Free Memory
│  ├─ free(A)
│  ├─ free(B)
│  └─ free(C)
│
└─ END: return 0
```

## Detailed Method Comparison Flow

```
                    GEMM Computation
                          │
                    ┌─────┴─────┐
                    │           │
              ┌─────▼────┐    ┌─▼──────────┐
              │  Serial  │    │ Parallel?  │
              │ 1 thread │    └─┬──────────┘
              └──────────┘      │
                                ├─ Yes ─┬─────────────────┐
                                │       │                 │
                            ┌───▼───┐  │             ┌────▼──────┐
                            │Shared?│  │             │Distributed│
                            └───┬───┘  │             │  Memory?  │
                                │      │             └────┬──────┘
                            ┌───▼───┐  │                  │
                            │OpenMP?│  │             ┌────▼──────┐
                            └───┬───┘  │             │  OpenCL?  │
                                │      │             └────┬──────┘
                        ┌───────┴──┐   │                  │
                        │CPU cores │   │        ┌─────────┘
                        │  threads │   │        │
                        └──────────┘   │    ┌───▼──────┐
                                       │    │GPU/CPU  │
                                       │    │fallback │
                                       │    └──────────┘
                                       │
                                    ┌──▼───────┐
                                    │Both CPUs │
                                    │  & GPUs? │
                                    └──┬───────┘
                                       │
                                   ┌───▼──────┐
                                   │  Hybrid  │
                                   │CPU+GPU   │
                                   │Partition │
                                   └──────────┘
```

## OpenMP Parallel Execution Model

```
Main Thread (tid=0)
│
├─ Fork: Create worker threads (tid=1..7)
│
├─ Parallel Region Start
│  │
│  ├─ Distribute work: (i,j) pairs to threads
│  │  ├─ Thread 0: rows i=0..127, columns j=0..511
│  │  ├─ Thread 1: rows i=128..255, columns j=0..511
│  │  ├─ Thread 2: rows i=256..383, columns j=0..511
│  │  ├─ ...
│  │  └─ Thread 7: rows i=896..1023, columns j=0..511
│  │
│  ├─ Each thread executes:
│  │  └─ for k in 0..1023:
│  │       C[i,j] += A[i,k] × B[k,j]
│  │
│  └─ Implicit Barrier: Threads wait for each other
│
├─ Join: Merge threads back to main
│
└─ Continue execution (single-threaded)
```

## OpenCL Kernel Execution Model

```
Host (CPU)
│
├─ Create context & device
├─ Allocate device memory (d_A, d_B, d_C)
├─ Transfer host data → device (A → d_A, B → d_B)
│
├─ Compile kernel (GPU-specific optimizations)
│
├─ Enqueue kernel:
│  │
│  └─ Device (GPU)
│     │
│     ├─ Work-item Grid: N × N work-items
│     │  (e.g., 1024 × 1024 = 1,048,576 work-items)
│     │
│     ├─ Work-items execute in parallel:
│     │  │
│     │  ├─ WI[0,0]: C[0,0] = sum(A[0,k] × B[k,0])
│     │  ├─ WI[0,1]: C[0,1] = sum(A[0,k] × B[k,1])
│     │  ├─ ...
│     │  └─ WI[1023,1023]: C[1023,1023] = sum(...)
│     │
│     └─ Synchronization barrier (hardware)
│
├─ Transfer result ← device (d_C → C)
│
└─ Cleanup (release device memory)
```

## Hybrid CPU+GPU Concurrent Execution

```
Time ──────────────────────────────────────────►

Main Thread
│
├─ Split workload:
│  ├─ CPU: rows 0 to N/2-1
│  └─ GPU: rows N/2 to N-1
│
├─ Launch:
│  │
│  ├──────────────────────────────────────────
│  │  OpenMP Thread Pool (CPU)
│  │  ├─ Thread 0: Compute C[0..127, :]
│  │  ├─ Thread 1: Compute C[128..255, :]
│  │  ├─ Thread 2: Compute C[256..383, :]
│  │  ├─ ...
│  │  └─ (implicit barrier at end)
│  │
│  ├──────────────────────────────────────────
│  │  GPU (OpenCL)
│  │  ├─ Kernel launches
│  │  ├─ Work-items: 512×1024 (GPU portion)
│  │  └─ Execute C[512..1023, :]
│  │
│  └──────────────────────────────────────────
│    Both running CONCURRENTLY!
│
├─ Join:
│  │
│  ├─ Wait for CPU (OpenMP barrier)
│  ├─ Wait for GPU (clFinish)
│  └─ Both complete before continuing
│
└─ Final result = CPU result + GPU result
```

## Performance Comparison Flow

```
Input: 4 timing measurements (serial, omp, ocl, hybrid)

┌──────────────────────────────────────┐
│ Calculate GFLOPs for each method     │
│ GFLOPs = (2×N³)/(10⁹ × time[s])      │
└──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ Calculate Speedup vs Serial          │
│ Speedup = time_serial / time_method  │
└──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ Calculate Efficiency                 │
│ Efficiency = Speedup / Num_Cores     │
└──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ Format & display results             │
│ [Table with all metrics]             │
└──────────────────────────────────────┘
```

