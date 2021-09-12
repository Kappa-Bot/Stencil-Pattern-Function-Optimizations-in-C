# Stencil-Rope-Optimizations
C program that calculates the movement equation of a 1-dimensional rope over time, follows an stencil computational pattern. Optimizations are implemented to mitigate specific bottlenecks.

## Compilation
..... I'll put it here when I get to remember it
### Usage
./Stencil.o [V] [N] [I] [T]
#### V stands for the Version of the program you want to execute
#### N stands for Number of elements on the rope to store in memory (Total of N + 2)
#### I stands for the number of time Instants to compute the equation over the rope
#### T stands for the number Threads you want to execute in one of the MultiThread version.

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
