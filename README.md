# Stencil-Rope-Optimizations
C program that calculates the movement equation of a 1-dimensional rope over time, follows an stencil computational pattern. Optimizations are implemented to mitigate specific bottlenecks.

## Compilation
..... I'll put it here when I get to remember it

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
