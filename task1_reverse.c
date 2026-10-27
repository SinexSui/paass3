#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_table(const char *title, char letter, int rank, int size)
{
    char *all = NULL;
    if (rank == 0) all = malloc(size);

    MPI_Gather(&letter, 1, MPI_CHAR, all, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("%s\n", title);
        printf("%-8s %s\n", "Process", "Element");
        for (int i = 0; i < size; i++)
            printf("%-8d \"%c\"\n", i, all[i]);
        printf("\n");
        free(all);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char word[] = "Parallelization";
    const int  len    = (int)strlen(word);

    if (size != len) {
        if (rank == 0)
            fprintf(stderr, "Нужно ровно %d процессов (по одному на букву слова \"%s\"), "
                            "а запущено %d.\n", len, word, size);
        MPI_Finalize();
        return 1;
    }

    char letter;
    MPI_Scatter(word, 1, MPI_CHAR, &letter, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    print_table("Initially", letter, rank, size);

    int  partner = size - 1 - rank;
    char new_letter;
    MPI_Sendrecv(&letter,     1, MPI_CHAR, partner, 0,
                 &new_letter, 1, MPI_CHAR, partner, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    print_table("Becomes", new_letter, rank, size);

    MPI_Finalize();
    return 0;
}
