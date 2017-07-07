#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <iostream>
//using namespace std;

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

        for (i = 0; i < n; ++i){
                double x = (double) rand() / RAND_MAX;
                double y = (double) rand() / RAND_MAX;
                if ((x * x + y * y) < 1){
                        sum ++;
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
                //int mod = N % size;
                //int weight = N / size;
                double ans = 0;
                int i = 0;
                for (i = 0; i < size; ++i)
                        ans += results[i];
                printf("Final Results = %.9f\n", 4.0 * ans / N);
        }

        MPI_Finalize();
}
