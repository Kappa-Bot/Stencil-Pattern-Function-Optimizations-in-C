# Stencil-Rope-Optimizations
C program that applies an equation of a 1-dimensional data buffer over time, which follows an stencil computational pattern. This work gives several approaches to mitigate specific bottlenecks through optimizations.

## Compilation
C Compiler: gcc
Base Flags: -O3 -lm
MultiThread: -fopenmp -fopenacc
GPU: -fopenacc
## Usage
./Stencil.o [V] [N] [I] [T]
#### (V)ersion of the program you want to execute
#### (N)umber of elements on the rope to store in memory (Total of N + 2)
#### (I)nstants amount in order to compute the equation over the data
#### (T)hreads to run on the program for the multithreaded version.

## Optimizations
### Double and Triple Buffer
Todo...
### Time Blocking (2-3-...)
Todo...
### Non-Temporal Memory Writing  
Todo...

## Parallelization
### OpenMP
Multi-Threaded and Multi-Core Execution of the program. Paralellized by time instants.
### OpenACC
GPU Execution of the program. Every time instant of the problem requires one migration to the device.
### CUDA
Still looking for that version of my program...

## Target
CPU: Intel Nehalem, 2 threads per core, 4 cores, 1 socket 
GPU: ... Pascal (don't totally remember)

## Results
... Reports + Graphics + etc. When I remember where did I put'em
