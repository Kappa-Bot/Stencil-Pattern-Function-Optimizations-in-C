#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REAL double

#define L (REAL) 0.16
#define L2 (REAL) (2.0 - 2.0 * L)

// For 2*N > size of LLC
void StencilNonTemporal(REAL *IN1, REAL *IN2, REAL *OUT, unsigned long N);
