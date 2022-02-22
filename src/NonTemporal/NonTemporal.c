///////////////////////////////////////////////////////////////

/**
 *      Stencil: Non Temporal Writes Optimization Code
 **/

///////////////////////////////////////////////////////////////

#include "NonTemporal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////

void StencilNonTemporal(REAL *IN1, REAL *IN2, REAL *OUT, unsigned long N) {
    #pragma vector nontemporal
    for (unsigned long i = 1; i < N; i++)
        OUT[i] = L2 * IN1[i]
                + L * (IN1[i + 1] + IN1[i - 1])
                - IN2[i];
}
