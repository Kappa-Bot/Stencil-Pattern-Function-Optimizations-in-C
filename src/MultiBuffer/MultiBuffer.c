///////////////////////////////////////////////////////////////

/**
 *          Stencil: Multiple Buffer Optimization Code
 **/

///////////////////////////////////////////////////////////////

#include "MultiBuffer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////

void StencilBuffer(REAL *IN1, REAL *IN2, REAL *OUT, unsigned long N) {
    for (unsigned long i = 1; i < N; i++)
        OUT[i] = L2 * IN1[i]
                + L * (IN1[i + 1] + IN1[i - 1])
                - IN2[i];
}

void StencilBufferOptimal(REAL *IN, REAL *OUT, unsigned long N) {
    for (unsigned long i = 1; i < N; i++)
        OUT[i] = (L2 * IN[i] - OUT[i])
                + L * (IN[i + 1] + IN[i - 1]);
}