///////////////////////////////////////////////////////////////

/**
 *                      Stencil Base Code
 **/

///////////////////////////////////////////////////////////////

#include "Stencil.h"
#include "MultiBuffer/MultiBuffer.c"
#include "NonTemporal/NonTemporal.c"
#include "TimeBlock/TimeBlock.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>
#include <openacc.h>

///////////////////////////////////////////////////////////////

__attribute__ ((noinline)) REAL CheckSum(REAL *DATA, unsigned long N) {
    REAL S = 0.0;
    #pragma omp for simd
    for (unsigned long i = 0; i < N + 1; i++)
        S = S + DATA[i];
    return S;
}

void Stencil(REAL **DATA, unsigned long I, unsigned long N) {
    for (unsigned long i = 1; i < N; i++)
        DATA[I + 1][i] = 2.0 * (1.0 - L) * DATA[I][i]
                        + L * (DATA[I][i + 1] + DATA[I][i - 1])
                        - DATA[I - 1][i];
}

void StencilOMP(REAL *IN, REAL *OUT, unsigned long N, unsigned long NTHR) {
    #pragma omp parallel for simd num_threads(NTHR)
    for (unsigned long i = 1; i < N; i++)
        OUT[i] = (L2 * IN[i] - OUT[i])
                + L * (IN[i + 1] + IN[i - 1]);
}

void StencilACC(REAL *IN, REAL *OUT, unsigned long N) {
    #pragma acc data present(IN, OUT)
    {
        #pragma acc parallel loop independent
        for (unsigned long i = 1; i < N; i++)
            OUT[i] = (L2 * IN[i] - OUT[i])
                    + L * (IN[i + 1] + IN[i - 1]);
    }
}

void StencilTriBlkOMP(REAL *IN1, REAL *IN2, REAL *OUT, REAL *NEW, unsigned long N, unsigned long NTHR) {
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

    #pragma omp parallel for simd num_threads(NTHR)
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

void StencilTriBlkNTOMP(REAL *restrict IN1, REAL *restrict IN2, REAL *restrict OUT, REAL *restrict NEW, unsigned long N, unsigned long NTHR) {
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

    #pragma omp parallel for simd num_threads(NTHR)
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

//#pragma acc routine
void StencilTriBlkACC(REAL *restrict IN1, REAL *restrict IN2, REAL *restrict OUT, REAL *restrict NEW, unsigned long N) {
    #pragma acc data present(IN1[0:N+1], IN2[0:N+1], OUT[0:N+1], NEW[0:N+1])
    {
        #pragma acc kernels
        {


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
        #pragma acc loop independent
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
    }
}