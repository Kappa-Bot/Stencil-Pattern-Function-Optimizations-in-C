#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <openacc.h>

#include "../src/Stencil.c"

int main(int argc, char **argv)
{
    int V = DEFAULT;
    int N = POINTS;
    int I = INSTANTS;
    int T = SINGLE;

    REAL Sum = 0.0;
    REAL *restrict A, *restrict B, *restrict C, *restrict D;
    int i, j;

    if (argc > 1) V = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) I = atoi(argv[3]);
    if (argc > 4) T = atoi(argv[4]);

    printf("Rope with %d points moving on %d instants\n", N + 1, I + 1);

    switch(V) {
        case 0: {
            printf("Original version\n");
            REAL **ROPE = (REAL **)malloc((I + 1) * sizeof(REAL*));
            for (i = 0; i < I + 1; i++) {
                ROPE[i] = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
                ROPE[i][0] = -1.0; //Position to start moving
                ROPE[i][N] = -1.0; //Position to start moving
            }

            for (j = 1; j < I; j++)
                Stencil(ROPE, j, N);

            Sum = CheckSum(ROPE[I], N);

            for(i = 0; i < I + 1; i++)
                free(ROPE[i]);
            break;
        }

        case 1: {
            printf("Triple Buffer version\n");

            A = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            B = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            REAL *C = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            A[0] = B[0] = C[0] = -1.0; //Position to start moving
            A[N] = B[N] = C[N] = -1.0; //Position to start moving

            for (j = 1; j <= I; j++)
                if (j % 3 == 1)
                    StencilBuffer(A, C, B, N);
                else if (j % 3 == 2)
                    StencilBuffer(B, A, C, N);
                else
                    StencilBuffer(C, B, A, N);

            Sum = CheckSum(j % 3 == 0 ? B : j % 3 == 1 ? C : A, N);

            printf("%f, %f, %f\n", A[1], B[1], C[1]);
            printf("%f, %f, %f\n", A[N - 1], B[N - 1], C[N - 1]);
            free(A); free(B); free(C);
            break;
        }

        case 2: {
            printf("Doble Buffer version\n");

            A = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            B = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            A[0] = B[0] = -1.0; //Position to start moving
            A[N] = B[N] = -1.0; //Position to start moving

            for (j = 1; j <= I; j++)
                StencilBufferOptimal((j % 2) == 1 ? A : B, (j % 2) == 1 ? B : A, N);

            Sum = CheckSum(j % 2 == 0 ? A : B, N);

            free(A); free(B);
            break;
        }

        case 3: {
            printf("Time block 3 buffer version\n");

            A = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            B = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            REAL *C = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            REAL *D = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            A[0] = B[0] = C[0] = D[0] = -1.0; //Position to start moving
            A[N] = B[N] = C[N] = D[N] = -1.0; //Position to start moving

            for (j = 1; j <= I; j+=2)
                if (j % 4 <= 1)
                    StencilTimeBlock(A, C, B, D, N);
                else
                    StencilTimeBlock(D, B, C, A, N);

            printf("%f, %f, %f, %f\n", A[1], B[1], C[1], D[1]);
            printf("%f, %f, %f, %f\n", A[N - 1], B[N - 1], C[N - 1], D[N - 1]);

            Sum = CheckSum(j % 3 == 0 ? A : j % 3 == 1 ? C : B, N);
            //Sum = CheckSum(B, N);

            free(A); free(B); free(C);
            break;
        }

        case 4: {
            printf("%d-Thread version of Doble Buffer\n", T);

            A = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            B = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            A[0] = B[0] = -1.0; //Position to start moving
            A[N] = B[N] = -1.0; //Position to start moving

            REAL *ROPE_LAST = ((I + 1) % 2 == 0) ? A : B;

            //#pragma omp parallel default(none) shared(Sum) private(i, j) firstprivate(N, I, A, B, ROPE) num_threads(T)
            {
                for (j = 1; j <= I; j++)
                    if (j % 2 == 1)
                        StencilOMP(A, B, N, T);
                    else
                        StencilOMP(B, A, N, T);

                //#pragma omp barrier
                #pragma omp parallel for simd reduction(+:Sum) num_threads(T)
                for (i = 0; i < N + 1; i++)
                    Sum += ROPE_LAST[i];
            }

            free(A); free(B);
            break;
        }

        case 5: {
            printf("GPU - OpenACC version\n");

            A = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            B = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            B = j % 2 == 1 ? A : B;
            A[0] = B[0] = -1.0; //Position to start moving
            A[N] = B[N] = -1.0; //Position to start moving

            #pragma acc data copyin(A[0:N+1], B[0:N+1])
            {
                        for (j = 1; j <= I; j++)
                            #pragma acc kernels
                            {
                                if (j % 2 == 1) {
                                    //#pragma acc parallel loop
                                    for (int i = 1; i < N; i++)
                                        B[i] = (L2 * A[i] - B[i])
                                                + L * (A[i + 1] + A[i - 1]);
                                }
                                else {
                                    //#pragma acc parallel loop
                                    for (int i = 1; i < N; i++)
                                        A[i] = (L2 * B[i] - A[i])
                                                + L * (B[i + 1] + B[i - 1]);
                                }
                            }
    //                        StencilACC((j % 2) == 1 ? A : B, (j % 2) == 1 ? B : A, N);
                    #pragma acc kernels
                        for (int i = 0; i < N + 1; i++)
                            Sum += B[i];
            }

            free(A); free(B);
            break;
        }

        case 6: {
            printf("Triple Buffer + NonTemporal Writes version\n");

            A = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            B = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            REAL *C = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            A[0] = B[0] = C[0] = -1.0; //Position to start moving
            A[N] = B[N] = C[N] = -1.0; //Position to start moving

            for (j = 1; j <= I; j++) {
                if (j % 3 == 1)
                    StencilNonTemporal(A, C, B, N);
                else if (j % 3 == 2)
                    StencilNonTemporal(B, A, C, N);
                else
                    StencilNonTemporal(C, B, A, N);
            }

            Sum = CheckSum(j == 0 ? B : j % 3 == 1 ? C : A, N);

            free(A); free(B); free(C);
            break;
        }

        case 7: {
            printf("Time block 3 buffer + Non temporal writes version\n");

            A = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            B = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            C = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            D = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            A[0] = B[0] = C[0] = D[0] = -1.0; //Position to start moving
            A[N] = B[N] = C[N] = D[N] = -1.0; //Position to start moving

            for (j = 1; j <= I; j+=2)
                if (j % 4 <= 1)
                    StencilTimeBlockNonTemporal(A, C, B, D, N);
                else
                    StencilTimeBlockNonTemporal(D, B, C, A, N);

            printf("%f, %f, %f, %f\n", A[1], B[1], C[1], D[1]);
            printf("%f, %f, %f, %f\n", A[N - 1], B[N - 1], C[N - 1], D[N - 1]);

            //TODO: Make this correct for all I input
            Sum = CheckSum(j % 3 == 0 ? A : j % 3 == 1 ? C : B, N);

            free(A); free(B); free(C);
            break;
        }

        case 8: {
            printf("%d-Thread version of Triple Time Block 4 Buffer\n", T);

            A = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            B = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            C = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            D = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            A[0] = B[0] = C[0] = D[0] = -1.0; //Position to start moving
            A[N] = B[N] = C[N] = D[N] = -1.0; //Position to start moving


            //#pragma omp parallel default(none) shared(Sum) private(i, j) firstprivate(N, I, A, B, C, D, ROPE) num_threads(T)
            {
                //TODO: Make this correct for all I input
                if (I % 3 != 0)
                    StencilBuffer(B, C, A, N);
                for (j = 1; j <= I; j += 3)
                    if (j % 6 <= 2)
                        StencilTriBlkOMP(A, C, B, D, N, T);
                    else
                        StencilTriBlkOMP(D, B, C, A, N, T);

                REAL *ROPE = B;
                //#pragma omp barrier
                #pragma omp parallel for simd reduction(+:Sum) num_threads(T)
                for (i = 0; i < N + 1; i++)
                    Sum += ROPE[i];
            }

            printf("%f, %f, %f, %f\n", A[1], B[1], C[1], D[1]);
            printf("%f, %f, %f, %f\n", A[N - 1], B[N - 1], C[N - 1], D[N - 1]);

            //Sum = CheckSum(j % 3 == 0 ? B : j % 3 == 1 ? C : A, N);

            free(A); free(B); free(C);
            break;
        }

        case 9: {
            printf("Triple time block 4 buffer version\n");

            A = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            B = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            C = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            D = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            A[0] = B[0] = C[0] = D[0] = -1.0; //Position to start moving
            A[N] = B[N] = C[N] = D[N] = -1.0; //Position to start moving
            printf("%d, %d\n", I / 3, I % 3);
            //TODO: Make this correct for all I input
            if (I % 3 != 0)
                StencilBuffer(B, C, A, N);
            for (j = 1; j <= I; j += 3)
                if (j % 6 <= 2)
                    StencilTimeBlock3(A, C, B, D, N);
                else
                    StencilTimeBlock3(D, B, C, A, N);

            printf("%f, %f, %f, %f\n", A[1], B[1], C[1], D[1]);
            printf("%f, %f, %f, %f\n", A[N - 1], B[N - 1], C[N - 1], D[N - 1]);

            Sum = CheckSum(B, N);

            free(A); free(B); free(C);
            break;
        }

        case 10: {
            printf("Triple time block 4 buffer + Non temporal writes version\n");

            A = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            B = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            C = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            D = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            A[0] = B[0] = C[0] = D[0] = -1.0; //Position to start moving
            A[N] = B[N] = C[N] = D[N] = -1.0; //Position to start moving
            printf("%d, %d\n", I / 3, I % 3);
            //TODO: Make this correct for all I input
            if (I % 3 != 0)
                StencilBuffer(B, C, A, N);
            for (j = 1; j <= I; j += 3)
                if (j % 6 <= 2)
                    StencilTimeBlock3NonTemporal(A, C, B, D, N);
                else
                    StencilTimeBlock3NonTemporal(D, B, C, A, N);

            printf("%f, %f, %f, %f\n", A[1], B[1], C[1], D[1]);
            printf("%f, %f, %f, %f\n", A[N - 1], B[N - 1], C[N - 1], D[N - 1]);

            Sum = CheckSum(B, N);

            free(A); free(B); free(C);
            break;
        }

        case 11: {
            printf("%d-Thread version of Triple Time Block 4 Buffer + Non temporal writes\n", T);

            A = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            B = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            C = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            D = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            A[0] = B[0] = C[0] = D[0] = -1.0; //Position to start moving
            A[N] = B[N] = C[N] = D[N] = -1.0; //Position to start moving


            //#pragma omp parallel default(none) shared(Sum) private(i, j) firstprivate(N, I, A, B, C, D, ROPE) num_threads(T)
            {
                //TODO: Make this correct for all I input
                if (I % 3 != 0)
                    StencilNonTemporal(B, C, A, N);
                for (j = 1; j <= I; j += 3)
                    if (j % 6 <= 2)
                        StencilTriBlkNTOMP(A, C, B, D, N, T);
                    else
                        StencilTriBlkNTOMP(D, B, C, A, N, T);

                REAL *ROPE = B;
                //#pragma omp barrier
                #pragma omp parallel for simd reduction(+:Sum) num_threads(T)
                for (i = 0; i < N + 1; i++)
                    Sum += ROPE[i];
            }

            printf("%f, %f, %f, %f\n", A[1], B[1], C[1], D[1]);
            printf("%f, %f, %f, %f\n", A[N - 1], B[N - 1], C[N - 1], D[N - 1]);

            //Sum = CheckSum(j % 3 == 0 ? B : j % 3 == 1 ? C : A, N);

            free(A); free(B); free(C);
            break;
        }

        case 12: {
            printf("GPU of Triple time block 4 buffer version\n");

            A = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            B = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            C = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            D = (REAL *restrict)malloc((N + 1) * sizeof(REAL));
            A[0] = B[0] = C[0] = D[0] = -1.0; //Position to start moving
            A[N] = B[N] = C[N] = D[N] = -1.0; //Position to start moving
            #pragma acc data copyin(A[0:N+1], B[0:N+1], C[0:N+1], D[0:N+1])
            {
                printf("%d, %d\n", I / 3, I % 3);
                //TODO: Make this correct for all I input
                for (j = 1; j <= I; j += 3)
                    if (j % 6 <= 2)
                        StencilTriBlkACC(A, C, B, D, N);
                    else
                        StencilTriBlkACC(D, B, C, A, N);

                printf("%f, %f, %f, %f\n", A[1], B[1], C[1], D[1]);
                printf("%f, %f, %f, %f\n", A[N - 1], B[N - 1], C[N - 1], D[N - 1]);
                #pragma acc kernels
                for (int i = 0; i < N + 1; i++)
                    Sum += B[i];
            }

            free(A); free(B); free(C);
            break;
        }

        default: {
            fprintf(stderr, "Error, available versions are [0 - 3]\n");
            exit(EXIT_FAILURE);
        }
    }

    printf("\nChecksum: %e\n", Sum);
    exit(EXIT_SUCCESS);
}
