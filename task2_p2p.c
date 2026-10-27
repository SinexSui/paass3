#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TAG_A = 1, TAG_B = 2, TAG_C = 3 };

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int gen = 0, reps = 1, show_time = 0, quiet = 0, n = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--gen")  && i + 1 < argc) { gen = 1; n = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) { reps = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--time"))  show_time = 1;
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
    }
    if (reps < 1) reps = 1;

    double *A = NULL, *B = NULL, *C = NULL;
    int ok = 1;

    if (rank == 0) {
        if (!gen && scanf("%d", &n) != 1) ok = 0;
        if (n < 1) ok = 0;
    }

    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            MPI_Send(&ok, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
            MPI_Send(&n,  1, MPI_INT, r, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&ok, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&n,  1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (!ok) {
        if (rank == 0) fprintf(stderr, "Некорректный ввод: ожидается n >= 1\n");
        MPI_Finalize();
        return 1;
    }

    const int rows   = (n + size - 1) / size;
    const int padded = rows * size;
    const int blk    = rows * n;

    if (rank == 0) {
        A = calloc((size_t)padded * n, sizeof(double));
        B = malloc((size_t)n * n * sizeof(double));
        C = calloc((size_t)padded * n, sizeof(double));
        if (gen) {
            srand(12345);
            for (long i = 0; i < (long)n * n; i++) { A[i] = rand() % 10; B[i] = rand() % 10; }
        } else {
            for (long i = 0; i < (long)n * n; i++) if (scanf("%lf", &A[i]) != 1) A[i] = 0;
            for (long i = 0; i < (long)n * n; i++) if (scanf("%lf", &B[i]) != 1) B[i] = 0;
        }
    }

    double *myA = (rank == 0) ? A : malloc((size_t)blk * sizeof(double));
    double *myC = (rank == 0) ? C : malloc((size_t)blk * sizeof(double));
    double *Bf  = (rank == 0) ? B : malloc((size_t)n * n * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int rep = 0; rep < reps; rep++) {

        if (rank == 0) {
            for (int r = 1; r < size; r++) {
                MPI_Send(A + (size_t)r * blk, blk, MPI_DOUBLE, r, TAG_A, MPI_COMM_WORLD);
                MPI_Send(B, n * n,              MPI_DOUBLE, r, TAG_B, MPI_COMM_WORLD);
            }

        } else {
            MPI_Recv(myA, blk,   MPI_DOUBLE, 0, TAG_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(Bf,  n * n, MPI_DOUBLE, 0, TAG_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double *out = (rank == 0) ? C : myC;
        memset(out, 0, (size_t)blk * sizeof(double));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < n; j++) {
                double a = myA[i * n + j];
                for (int m = 0; m < n; m++)
                    out[i * n + m] += a * Bf[j * n + m];
            }

        if (rank == 0) {
            for (int r = 1; r < size; r++)
                MPI_Recv(C + (size_t)r * blk, blk, MPI_DOUBLE, r, TAG_C,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Send(myC, blk, MPI_DOUBLE, 0, TAG_C, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = (MPI_Wtime() - t0) / reps;

    if (rank == 0) {
        if (!quiet)
            for (int i = 0; i < n; i++) {
                for (int m = 0; m < n; m++)
                    printf(m ? " %.10g" : "%.10g", C[i * n + m]);
                printf("\n");
            }
        if (show_time)
            fprintf(stderr, "[p2p]        n=%d procs=%d reps=%d  time/rep = %.9f s\n",
                    n, size, reps, elapsed);
    }

    if (rank != 0) { free(myA); free(myC); free(Bf); }
    if (rank == 0) { free(A); free(B); free(C); }
    MPI_Finalize();
    return 0;
}
