#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <openacc.h>

#define REAL double
#define POINTS 5000000 //5MB
#define INSTANTS 1000 //1K
#define DEFAULT 0 //NO CHANGES
#define SINGLE 1 //1THR
#define L (REAL) 0.16
#define L2 (REAL) (2.0 - 2.0 * L)

//Verify data integrity
__attribute__ ((noinline)) REAL CheckSum(REAL *ROPE, int N) {
    REAL S = 0.0;
    #pragma omp for simd
    for (int i = 0; i < N + 1; i++)
        S = S + ROPE[i];
    return S;
}

void StencilORG(REAL **ROPE, int I, int N) {
    for (int i = 1; i < N; i++)
        ROPE[I + 1][i] = 2.0 * (1.0 - L) * ROPE[I][i]
                        + L * (ROPE[I][i + 1] + ROPE[I][i - 1])
                        - ROPE[I - 1][i];
}

//Swap the 3 functions simultaneously so we'll create only these
void StencilBuff(REAL *IN1, REAL *IN2, REAL *OUT, int N) {
    for (int i = 1; i < N; i++)
        OUT[i] = L2 * IN1[i]
                + L * (IN1[i + 1] + IN1[i - 1])
                - IN2[i];
}

//For 2*N > size of LLC
void StencilNT(REAL *IN1, REAL *IN2, REAL *OUT, int N) {
    #pragma vector nontemporal
    for (int i = 1; i < N; i++)
        OUT[i] = L2 * IN1[i]
                + L * (IN1[i + 1] + IN1[i - 1])
                - IN2[i];
}

//OUT acts as prev and new instants at the same time
void StencilOPT(REAL *IN, REAL *OUT, int N) {
    for (int i = 1; i < N; i++)
        OUT[i] = (L2 * IN[i] - OUT[i])
                + L * (IN[i + 1] + IN[i - 1]);
}

void StencilOMP(REAL *IN, REAL *OUT, int N, int NTHR) {
    #pragma omp parallel for simd num_threads(NTHR)
    for (int i = 1; i < N; i++)
        OUT[i] = (L2 * IN[i] - OUT[i])
                + L * (IN[i + 1] + IN[i - 1]);
}

void StencilACC(REAL *IN, REAL *OUT, int N) {
    #pragma acc data present(IN, OUT)
    {
        #pragma acc loop independent
        for (int i = 1; i < N; i++)
            OUT[i] = (L2 * IN[i] - OUT[i])
                    + L * (IN[i + 1] + IN[i - 1]);
    }
}

#pragma omp declare simd aligned(IN:32)
REAL ApplyStencil(REAL *IN, REAL *OUT, int INDEX) {
    return L2 * IN[INDEX] + L * (IN[INDEX + 1] + IN[INDEX - 1]) - OUT[INDEX];
}

#define L2L2 (REAL)(L2 * L2)
#define LL2 (REAL)(L * L2)
#define LL (REAL)(L * L)

//Two movements at the same time, applies equation to every base element
void StencilBlock(REAL *IN1, REAL *IN2, REAL *OUT, REAL *NEW, int N) {
    /*
    //Sequential idea
    for (int i = 1; i < N; i++)
        OUT[i] = L2 * IN1[i]
                + L * (IN1[i + 1] + IN1[i - 1])
                - IN2[i];
    for (int i = 1; i < N; i++) {
        IN2[i] = L2 * OUT[i]
                + L * (OUT[i + 1] + OUT[i - 1])
                - IN1[i];
    }
    */
    REAL Left, Mid, Right, Prev;

    Left = -1.0;
    Mid = OUT[1] = L2 * IN1[1] + L * (-1.0 + IN1[2]) - IN2[1];
    Right = L2 * IN1[2] + L * (IN1[1] + IN1[3]) - IN2[2];
    NEW[1] = L2 * Mid + L * (Left + Right) - IN1[1];

    for (int i = 2; i < N - 1; i++) {
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

void StencilTriBlk(REAL *IN1, REAL *IN2, REAL *OUT, REAL *NEW, int N) {
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

    for (int i = 3; i < N - 2; i++) {
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

void StencilTriBlkNT(REAL *restrict IN1, REAL *restrict IN2, REAL *restrict OUT, REAL *restrict NEW, int N) {
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

    for (int i = 3; i < N - 2; i++) {
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

void StencilBlkNT(REAL *restrict IN1, REAL *restrict IN2, REAL *restrict OUT, REAL *restrict NEW, const int N) {
    REAL Left, Mid, Right, Prev;
    Left = -1.0;
    #pragma vector nontemporal
    Mid = OUT[1] = L2 * IN1[1] + L * (IN1[2] - 1.0) - IN2[1];
    Right = L2 * IN1[2] + L * (IN1[1] + IN1[3]) - IN2[2];
    #pragma vector nontemporal
    NEW[1] = L2 * Mid + L * (Left + Right) - IN1[1];

    #pragma vector nontemporal
    for (int i = 2; i < N - 1; i++) {
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

void StencilTriBlkOMP(REAL *IN1, REAL *IN2, REAL *OUT, REAL *NEW, int N, int NTHR) {
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
    for (int i = 3; i < N - 2; i++) {
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

void StencilTriBlkNTOMP(REAL *restrict IN1, REAL *restrict IN2, REAL *restrict OUT, REAL *restrict NEW, int N, int NTHR) {
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
    for (int i = 3; i < N - 2; i++) {
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
inline void StencilTriBlkACC(REAL *restrict IN1, REAL *restrict IN2, REAL *restrict OUT, REAL *restrict NEW, int N) {
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
        for (int i = 3; i < N - 2; i++) {
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