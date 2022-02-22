///////////////////////////////////////////////////////////////

/**
 *          Stencil: Time Blocking Optimization Code
 **/

///////////////////////////////////////////////////////////////

#include "TimeBlock.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////

/**
 * This function is functionally equivalent as 
 * executing the following 2 loops:
 *
    for (unsigned long i = 1; i < N + 1; i++)
        OUT[i] = L2 * IN1[i]
                + L * (IN1[i + 1] + IN1[i - 1])
                - IN2[i];
    for (unsigned long i = 1; i < N + 1; i++)
        IN2[i] = L2 * OUT[i]
                + L * (OUT[i + 1] + OUT[i - 1])
                - IN1[i];
 **/
void StencilTimeBlock(REAL *IN1, REAL *IN2, REAL *OUT, REAL *NEW, unsigned long N) {
    REAL Left, Mid, Right, Prev;

    Left = -1.0;
    Mid = OUT[1] = L2 * IN1[1] + L * (-1.0 + IN1[2]) - IN2[1];
    Right = L2 * IN1[2] + L * (IN1[1] + IN1[3]) - IN2[2];
    NEW[1] = L2 * Mid + L * (Left + Right) - IN1[1];

    for (unsigned long i = 2; i < N - 1; i++) {
        Left = L2 * IN1[i - 1] + L * (IN1[i] + IN1[i - 2]) - IN2[i - 1];
        Mid = OUT[i] = L2 * IN1[i] + L * (IN1[i + 1] + IN1[i - 1]) - IN2[i];
        Right = L2 * IN1[i + 1] + L * (IN1[i + 2] + IN1[i]) - IN2[i + 1];
        NEW[i] = L2 * Mid + L * (Left + Right) - IN1[i];
    }

    Left = L2 * IN1[N - 2] + L * (IN1[N - 3] + IN1[N - 1]) - IN2[N - 2];
    Mid = OUT[N - 1] = L2 * IN1[N - 1] + L * (IN1[N - 2] - 1.0) - IN2[N - 1];
    Right = -1.0;
    NEW[N - 1] = L2 * Mid + L * (Left + Right) - IN1[N - 1];
}


/**
 * This function is functionally equivalent as 
 * executing the following 3 loops:
 *
    for (unsigned long i = 1; i < N + 1; i++)
        OUT[i] = L2 * IN1[i]
                + L * (IN1[i + 1] + IN1[i - 1])
                - IN2[i];
    for (unsigned long i = 1; i < N + 1; i++)
        IN2[i] = L2 * OUT[i]
                + L * (OUT[i + 1] + OUT[i - 1])
                - IN1[i];
    for (unsigned long i = 1; i < N + 1; i++)
        IN1[i] = L2 * IN2[i]
                + L * (IN2[i + 1] + IN2[i - 1])
                - OUT[i];
 **/
void StencilTimeBlock3(REAL *IN1, REAL *IN2, REAL *OUT, REAL *NEW, unsigned long N) {
    REAL Left, Mid, Right, AUX1, AUX2, AUX3, AUX4, AUX5;

    AUX3 = L2 * IN1[1] + L * (-1.0 + IN1[2]) - IN2[1];
    AUX4 = L2 * IN1[2] + L * (IN1[1] + IN1[3]) - IN2[2];
    AUX5 = L2 * IN1[3] + L * (IN1[2] + IN1[4]) - IN2[3];
    Left = -1.0;
    Mid = OUT[1] = L2 * AUX3 + L * (-1.0 + AUX4) - IN1[1];
    Right = L2 * AUX4 + L * (AUX3 + AUX5) - IN1[2];
    NEW[1] = L2 * Mid + L * (Left + Right) - AUX3;

    AUX2 = L2 * IN1[1] + L * (-1.0 + IN1[2]) - IN2[1];
    AUX3 = L2 * IN1[2] + L * (IN1[1] + IN1[3]) - IN2[2];
    AUX4 = L2 * IN1[3] + L * (IN1[2] + IN1[4]) - IN2[3];
    AUX5 = L2 * IN1[4] + L * (IN1[3] + IN1[5]) - IN2[4];
    Left = L2 * AUX2 + L * (-1.0 + AUX3) - IN1[1];
    Mid = OUT[2] = L2 * AUX3 + L * (AUX2 + AUX4) - IN1[2];
    Right = L2 * AUX4 + L * (AUX3 + AUX5) - IN1[3];
    NEW[2] = L2 * Mid + L * (Left + Right) - AUX3;

    for (unsigned long i = 3; i < N - 2; i++) {
        AUX1 = L2 * IN1[i - 2] + L * (IN1[i - 1] + IN1[i - 3]) - IN2[i - 2];
        AUX2 = L2 * IN1[i - 1] + L * (IN1[i] + IN1[i - 2]) - IN2[i - 1];
        AUX3 = L2 * IN1[i] + L * (IN1[i + 1] + IN1[i - 1]) - IN2[i];
        AUX4 = L2 * IN1[i + 1] + L * (IN1[i + 2] + IN1[i]) - IN2[i + 1];
        AUX5 = L2 * IN1[i + 2] + L * (IN1[i + 1] + IN1[i + 3]) - IN2[i + 2];
        Left = L2 * AUX2 + L * (AUX1 + AUX3) - IN1[i - 1];
        Mid = OUT[i] = L2 * AUX3 + L * (AUX2 + AUX4) - IN1[i];
        Right = L2 * AUX4 + L * (AUX3 + AUX5) - IN1[i + 1];
        NEW[i] = L2 * Mid + L * (Left + Right) - AUX3;
    }

    AUX1 = L2 * IN1[N - 4] + L * (IN1[N - 3] + IN1[N - 5]) - IN2[N - 4];
    AUX2 = L2 * IN1[N - 3] + L * (IN1[N - 2] + IN1[N - 4]) - IN2[N - 3];
    AUX3 = L2 * IN1[N - 2] + L * (IN1[N - 1] + IN1[N - 3]) - IN2[N - 2];
    AUX4 = L2 * IN1[N - 1] + L * (IN1[N - 2] - 1.0) - IN2[N - 1];
    Left = L2 * AUX2 + L * (AUX1 + AUX3) - IN1[N - 3];
    Mid = OUT[N - 2] = L2 * AUX3 + L * (AUX2 + AUX4) - IN1[N - 2];
    Right = L2 * AUX4 + L * (AUX3 - 1.0) - IN1[N - 1];
    NEW[N - 2] = L2 * Mid + L * (Left + Right) - AUX3;

    AUX1 = L2 * IN1[N - 3] + L * (IN1[N - 2] + IN1[N - 4]) - IN2[N - 3];
    AUX2 = L2 * IN1[N - 2] + L * (IN1[N - 1] + IN1[N - 3]) - IN2[N - 2];
    AUX3 = L2 * IN1[N - 1] + L * (IN1[N - 2] - 1.0) - IN2[N - 1];
    Left = L2 * AUX2 + L * (AUX1 + AUX3) - IN1[N - 2];
    Mid = OUT[N - 1] = L2 * AUX3 + L * (AUX2 - 1.0) - IN1[N - 1];
    Right = -1.0;
    NEW[N - 1] = L2 * Mid + L * (Left + Right) - AUX3;
}

void StencilTimeBlockNonTemporal(REAL *restrict IN1, REAL *restrict IN2, REAL *restrict OUT, REAL *restrict NEW, const unsigned long N) {
    REAL Left, Mid, Right, Prev;
    Left = -1.0;
    #pragma vector nontemporal
    Mid = OUT[1] = L2 * IN1[1] + L * (IN1[2] - 1.0) - IN2[1];
    Right = L2 * IN1[2] + L * (IN1[1] + IN1[3]) - IN2[2];
    #pragma vector nontemporal
    NEW[1] = L2 * Mid + L * (Left + Right) - IN1[1];

    #pragma vector nontemporal
    for (unsigned long i = 2; i < N - 1; i++) {
        Left = L2 * IN1[i - 1] + L * (IN1[i] + IN1[i - 2]) - IN2[i - 1];
        #pragma vector nontemporal
        Mid = OUT[i] = L2 * IN1[i] + L * (IN1[i + 1] + IN1[i - 1]) - IN2[i];
        Right = L2 * IN1[i + 1] + L * (IN1[i + 2] + IN1[i]) - IN2[i + 1];
        #pragma vector nontemporal
        NEW[i] = L2 * Mid + L * (Left + Right) - IN1[i];
    }

    Left = L2 * IN1[N - 2] + L * (IN1[N - 3] + IN1[N - 1]) - IN2[N - 2];
    #pragma vector nontemporal
    Mid = OUT[N - 1] = L2 * IN1[N - 1] + L * (IN1[N - 2] - 1.0) - IN2[N - 1];
    Right = -1.0;
    #pragma vector nontemporal
    NEW[N - 1] = L2 * Mid + L * (Left + Right) - IN1[N - 1];
}

void StencilTimeBlock3NonTemporal(REAL *restrict IN1, REAL *restrict IN2, REAL *restrict OUT, REAL *restrict NEW, unsigned long N) {
    #pragma vector nontemporal
    {

    REAL Left, Mid, Right, AUX1, AUX2, AUX3, AUX4, AUX5;

    AUX3 = L2 * IN1[1] + L * (-1.0 + IN1[2]) - IN2[1];
    AUX4 = L2 * IN1[2] + L * (IN1[1] + IN1[3]) - IN2[2];
    AUX5 = L2 * IN1[3] + L * (IN1[2] + IN1[4]) - IN2[3];
    Left = -1.0;
    #pragma vector nontemporal
    Mid = OUT[1] = L2 * AUX3 + L * (-1.0 + AUX4) - IN1[1];
    Right = L2 * AUX4 + L * (AUX3 + AUX5) - IN1[2];
    #pragma vector nontemporal
    NEW[1] = L2 * Mid + L * (Left + Right) - AUX3;

    AUX2 = L2 * IN1[1] + L * (-1.0 + IN1[2]) - IN2[1];
    AUX3 = L2 * IN1[2] + L * (IN1[1] + IN1[3]) - IN2[2];
    AUX4 = L2 * IN1[3] + L * (IN1[2] + IN1[4]) - IN2[3];
    AUX5 = L2 * IN1[4] + L * (IN1[3] + IN1[5]) - IN2[4];
    Left = L2 * AUX2 + L * (-1.0 + AUX3) - IN1[1];
    #pragma vector nontemporal
    Mid = OUT[2] = L2 * AUX3 + L * (AUX2 + AUX4) - IN1[2];
    Right = L2 * AUX4 + L * (AUX3 + AUX5) - IN1[3];
    #pragma vector nontemporal
    NEW[2] = L2 * Mid + L * (Left + Right) - AUX3;

    for (unsigned long i = 3; i < N - 2; i++) {
        AUX1 = L2 * IN1[i - 2] + L * (IN1[i - 1] + IN1[i - 3]) - IN2[i - 2];
        AUX2 = L2 * IN1[i - 1] + L * (IN1[i] + IN1[i - 2]) - IN2[i - 1];
        AUX3 = L2 * IN1[i] + L * (IN1[i + 1] + IN1[i - 1]) - IN2[i];
        AUX4 = L2 * IN1[i + 1] + L * (IN1[i + 2] + IN1[i]) - IN2[i + 1];
        AUX5 = L2 * IN1[i + 2] + L * (IN1[i + 1] + IN1[i + 3]) - IN2[i + 2];
        Left = L2 * AUX2 + L * (AUX1 + AUX3) - IN1[i - 1];
        #pragma vector nontemporal
        Mid = OUT[i] = L2 * AUX3 + L * (AUX2 + AUX4) - IN1[i];
        Right = L2 * AUX4 + L * (AUX3 + AUX5) - IN1[i + 1];
        #pragma vector nontemporal
        NEW[i] = L2 * Mid + L * (Left + Right) - AUX3;
    }

    AUX1 = L2 * IN1[N - 4] + L * (IN1[N - 3] + IN1[N - 5]) - IN2[N - 4];
    AUX2 = L2 * IN1[N - 3] + L * (IN1[N - 2] + IN1[N - 4]) - IN2[N - 3];
    AUX3 = L2 * IN1[N - 2] + L * (IN1[N - 1] + IN1[N - 3]) - IN2[N - 2];
    AUX4 = L2 * IN1[N - 1] + L * (IN1[N - 2] - 1.0) - IN2[N - 1];
    Left = L2 * AUX2 + L * (AUX1 + AUX3) - IN1[N - 3];
    #pragma vector nontemporal
    Mid = OUT[N - 2] = L2 * AUX3 + L * (AUX2 + AUX4) - IN1[N - 2];
    Right = L2 * AUX4 + L * (AUX3 - 1.0) - IN1[N - 1];
    #pragma vector nontemporal
    NEW[N - 2] = L2 * Mid + L * (Left + Right) - AUX3;

    AUX1 = L2 * IN1[N - 3] + L * (IN1[N - 2] + IN1[N - 4]) - IN2[N - 3];
    AUX2 = L2 * IN1[N - 2] + L * (IN1[N - 1] + IN1[N - 3]) - IN2[N - 2];
    AUX3 = L2 * IN1[N - 1] + L * (IN1[N - 2] - 1.0) - IN2[N - 1];
    Left = L2 * AUX2 + L * (AUX1 + AUX3) - IN1[N - 2];
    #pragma vector nontemporal
    Mid = OUT[N - 1] = L2 * AUX3 + L * (AUX2 - 1.0) - IN1[N - 1];
    Right = -1.0;
    #pragma vector nontemporal
    NEW[N - 1] = L2 * Mid + L * (Left + Right) - AUX3;
    }
}
