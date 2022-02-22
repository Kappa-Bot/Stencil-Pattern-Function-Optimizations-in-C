#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REAL double

#define L (REAL) 0.16
#define L2 (REAL) (2.0 - 2.0 * L)

/**
 * Uses 3 always-in-memory buffers simultaneously, allowing to
 * rotate each ones role through the different instants,
 * avoiding to create, store, and operate within I data buffers.
 **/
void StencilBuffer(REAL *IN1, REAL *IN2, REAL *OUT, unsigned long N);

/**
 * Currently, best optimal version for single-thread
 * execution on most cases.
 *
 * OUT buffer acts as previous and next instants at the same time
 **/
void StencilBufferOptimal(REAL *IN, REAL *OUT, unsigned long N);
