#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>
#include <openacc.h>

#define POINTS 5000000 //5MB
#define INSTANTS 1000 //1K
#define DEFAULT 0 //NO CHANGES
#define SINGLE 1 //1THR

#define REAL double

#define L (REAL) 0.16
#define L2 (REAL) (2.0 - 2.0 * L)

__attribute__ ((noinline)) REAL CheckSum(REAL *ROPE, unsigned long N);                        // Verify data unsigned longegrity. Forces branch

void Stencil(REAL **ROPE, unsigned long I, unsigned long N);                                            // Original version of the program

void StencilOMP(REAL *IN, REAL *OUT, unsigned long N, unsigned long NTHR);                              // First proposal using OpenMP
void StencilTriBlkOMP(REAL *IN1, REAL *IN2, REAL *OUT, REAL *NEW, unsigned long N, unsigned long NTHR);
void StencilTriBlkNTOMP(REAL *restrict IN1, REAL *restrict IN2, REAL *restrict OUT, REAL *restrict NEW, unsigned long N, unsigned long NTHR);

void StencilACC(REAL *IN, REAL *OUT, unsigned long N);                                        // First proposal using OpenACC
//#pragma acc routine
void StencilTriBlkACC(REAL *restrict IN1, REAL *restrict IN2, REAL *restrict OUT, REAL *restrict NEW, unsigned long N);