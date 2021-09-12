# Stencil-Rope-Optimizations
C program that calculates the movement equation of a 1-dimensional rope over time, follows an stencil computational pattern. Optimizations are implemented to mitigate specific bottlenecks.

##Optimizations
###Double and Triple Buffer
Todo...
###Time Blocking (2-3-...)
Todo...
###Non-Temporal Memory Writing  
Todo...

##OpenMP
Multi-Threaded and Multi-Core Execution of the program. Paralellized by time instants.
##OpenACC
GPU Execution of the program. Every time instant of the problem requires one migration to the device.
##CUDA
Still looking for that version of my program...
