# 🏗️ System Architecture

Diagram ini menunjukkan arsitektur heterogeneous computing system untuk GEMM implementation.

## Hardware Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Host System (Windows)                    │
│                      Intel i7-12700 CPU                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         CPU (8 P-cores + 4 E-cores @ 3.6 GHz)            │  │
│  │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐      │  │
│  │  │C0 │ │C1 │ │C2 │ │C3 │ │C4 │ │C5 │ │C6 │ │C7 │      │  │
│  │  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘      │  │
│  │   P-cores (8x, high frequency)                          │  │
│  │  ┌───┐ ┌───┐ ┌───┐ ┌───┐                               │  │
│  │  │C8 │ │C9 │ │C10│ │C11│                               │  │
│  │  └───┘ └───┘ └───┘ └───┘                               │  │
│  │   E-cores (4x, efficient, low frequency)                │  │
│  │                                                         │  │
│  │         L3 Cache (20 MB shared)                         │  │
│  │         Memory Controller                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↕ (64-bit system bus)                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Main Memory (16 GB DDR4 3200 MHz)                │  │
│  │         Bandwidth: ~51 GB/s (theoretical)                │  │
│  │                                                         │  │
│  │    A[N×N]          B[N×N]          C[N×N]              │  │
│  │    (input)         (input)        (output)              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↕ (PCIe 3.0 x16 - 15 GB/s)             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   Integrated GPU (if available) / Discrete GPU           │  │
│  │   [Optional accelerator for OpenCL]                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Data Movement Flow:
1. Initialize: A, B → Main Memory
2. Execute:
   - OpenMP: Threads work directly on main memory
   - OpenCL: Copy data to GPU, compute, copy back
   - Hybrid: Partition work, CPU uses memory, GPU uses VRAM
3. Store: C ← Main Memory
```

## Software Stack Architecture

```
┌────────────────────────────────────────────────────────────┐
│              Application Layer (Benchmark)                 │
│  main.c: Measurement, timing, GFLOPs calculation         │
└────────────┬─────────────────────────────────────────────┘
             │
┌────────────┴──────────────────────────────────────────────┐
│          Implementation Layer (4 Methods)                  │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  1. Serial GEMM          (gemm_serial.c)            │ │
│  │     └─ Single-threaded baseline computation         │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  2. OpenMP GEMM          (gemm_openmp.c)            │ │
│  │     └─ Multi-threaded CPU parallelism               │ │
│  │     └─ Shared-memory model                          │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  3. OpenCL GEMM          (gemm_opencl.c)            │ │
│  │     └─ GPU/CPU compute                              │ │
│  │     └─ Distributed memory model                     │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  4. Hybrid GEMM          (gemm_hybrid.c)            │ │
│  │     └─ CPU + GPU concurrent                         │ │
│  │     └─ Row-based partitioning                       │ │
│  └─────────────────────────────────────────────────────┘ │
└────────────┬──────────────────────────────────────────────┘
             │
┌────────────┴──────────────────────────────────────────────┐
│         Runtime Layer (Parallelization APIs)              │
│                                                           │
│  OpenMP Runtime (libgomp)     OpenCL Runtime             │
│  ├─ Thread pool management    ├─ Device management      │
│  ├─ Work distribution         ├─ Kernel compilation     │
│  ├─ Barrier synchronization   ├─ Memory management      │
│  └─ Cache coherency           └─ Command queue          │
└────────────┬──────────────────────────────────────────────┘
             │
┌────────────┴──────────────────────────────────────────────┐
│        Hardware Layer (CPU Cores + GPU)                   │
│                                                           │
│  CPU Cores (OpenMP threads)   GPU Units (OpenCL work)    │
│  ├─ Execution                 ├─ Execution              │
│  ├─ Caches                    ├─ VRAM                   │
│  └─ Memory interconnect        └─ Compute Units         │
└────────────────────────────────────────────────────────────┘
```

## Data Structure Layout

```
For Matrix Size N=1024:

Each matrix: 1024 × 1024 floats = 4 MB

A[N×N]                   B[N×N]                   C[N×N]
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Row 0:        │       │ Row 0:        │       │ Row 0:        │
│ A[0,0..1023]  │       │ B[0,0..1023]  │       │ C[0,0..1023]  │
├───────────────┤       ├───────────────┤       ├───────────────┤
│ Row 1:        │       │ Row 1:        │       │ Row 1:        │
│ A[1,0..1023]  │       │ B[1,0..1023]  │       │ C[1,0..1023]  │
├───────────────┤       ├───────────────┤       ├───────────────┤
│ ...           │       │ ...           │       │ ...           │
├───────────────┤       ├───────────────┤       ├───────────────┤
│ Row 1023:     │       │ Row 1023:     │       │ Row 1023:     │
│ A[1023,...]   │       │ B[1023,...]   │       │ C[1023,...]   │
└───────────────┘       └───────────────┘       └───────────────┘

For Hybrid GEMM (50-50 split):

CPU computes:
┌───────────────────────────┐
│ C[0..511, 0..1023]        │ ← OpenMP threads
└───────────────────────────┘

GPU computes:
┌───────────────────────────┐
│ C[512..1023, 0..1023]     │ ← OpenCL kernels
└───────────────────────────┘

Then: CPU result + GPU result = Full C matrix
```

