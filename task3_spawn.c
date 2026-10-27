#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void ordered_print(MPI_Comm comm, int parent_rank)
{
    int rank, size, token = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank > 0)
        MPI_Recv(&token, 1, MPI_INT, rank - 1, 99, comm, MPI_STATUS_IGNORE);

    if (parent_rank < 0)
        printf("I am %d process from %d processes! My parent is none.\n", rank, size);
    else
        printf("I am %d process from %d processes! My parent is %d.\n", rank, size, parent_rank);
    fflush(stdout);

    if (rank < size - 1)
        MPI_Send(&token, 1, MPI_INT, rank + 1, 99, comm);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm parent;
    MPI_Comm_get_parent(&parent);

    if (parent == MPI_COMM_NULL) {

        int rank, n = 0, ok = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0 && (scanf("%d", &n) != 1 || n < 1)) {
            fprintf(stderr, "Некорректный ввод: ожидается целое n >= 1\n");
            ok = 0;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ok) { MPI_Finalize(); return 1; }

        ordered_print(MPI_COMM_WORLD, -1);
        MPI_Barrier(MPI_COMM_WORLD);

        char  root_arg[16];
        snprintf(root_arg, sizeof root_arg, "%d", 0);
        char *child_argv[] = { root_arg, NULL };

        MPI_Comm inter;
        MPI_Comm_spawn(argv[0], child_argv, n, MPI_INFO_NULL, 0,
                       MPI_COMM_WORLD, &inter, MPI_ERRCODES_IGNORE);

        MPI_Comm_disconnect(&inter);
    } else {

        int parent_rank = (argc > 1) ? atoi(argv[1]) : 0;
        ordered_print(MPI_COMM_WORLD, parent_rank);
        MPI_Comm_disconnect(&parent);
    }

    MPI_Finalize();
    return 0;
}
