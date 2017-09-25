/*
mpicc pi_mpi+omp.c -o pi_mpi+omp -fopenmp
time mpirun -np 3 ./pi_mpi+omp 100000000 4 20
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main(int argc, char** argv){
        int rank, size;

        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);

        int N = atoi(argv[1]);
        srand(time(NULL) + rank);

        int n = N / size;
        if (rank < N % size) n++;

        int sum = 0;
        int i = 0;
        int num_threads = atoi(argv[2]);
        int chunk = atoi(argv[3]);

        double x, y;

        #pragma omp parallel num_threads(num_threads) reduction(+:sum) private(x, y)
        {
                #pragma omp for schedule(dynamic, chunk)
                for (i = 0; i < n; ++i){
                        x = (double) rand_r(i) / RAND_MAX;
                        y = (double) rand_r(i) / RAND_MAX;
                        if ((x * x + y * y) < 1){
                                sum ++;
                        }
                }
        }

        double cur_res[1] = {(double) sum};

        double *results = (double*)malloc(sizeof(double)*size);

        MPI_Datatype rtype;
        MPI_Type_contiguous(1, MPI_DOUBLE, &rtype);
        MPI_Type_commit(&rtype);
        MPI_Gather(cur_res, 1, MPI_DOUBLE, results, 1, rtype,
                0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        printf("Results of node %d : %.9f\n", rank, 4.0 * sum / n);
        if (rank == 0){
                double ans = 0;
                int i = 0;
                for (i = 0; i < size; ++i)
                        ans += results[i];
                printf("Final Results = %.9f\n", 4.0 * ans / N);
        }

        MPI_Finalize();
}