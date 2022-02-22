#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REAL double

#define L (REAL) 0.16
#define L2 (REAL) (2.0 - 2.0 * L)

#define L2L2 (REAL)(L2 * L2)
#define LL2 (REAL)(L * L2)
#define LL (REAL)(L * L)

// Two applications of the equation at the same time
void StencilTimeBlock(REAL *IN1, REAL *IN2, REAL *OUT, REAL *NEW, unsigned long N);

// Three applications of the equation at the same time
void StencilTimeBlock3(REAL *IN1, REAL *IN2, REAL *OUT, REAL *NEW, unsigned long N);

void StencilTimeBlockNonTemporal(REAL *restrict IN1, REAL *restrict IN2, REAL *restrict OUT, REAL *restrict NEW, const unsigned long N);

void StencilTimeBlock3NonTemporal(REAL *restrict IN1, REAL *restrict IN2, REAL *restrict OUT, REAL *restrict NEW, unsigned long N);