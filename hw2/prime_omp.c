/*
gcc prime_omp.c -o prime_omp -fopenmp
gcc-5 prime_omp.c -o prime_omp -fopenmp

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int isPrime(int i){
    int sqrt_i = (int)sqrt((double)i);
    int j;
    for(j = 3; j <= sqrt_i; j += 2){
        if(i % j == 0)
            return 0;
    }
    return 1;
}

int main(int argc, char** argv){
    int i, j;
    int N = 50000;
    int count = 0;
    int flag;
    int num_threads = atoi(argv[1]);
    int chunk = atoi(argv[2]);

    #pragma omp parallel num_threads(num_threads) reduction(+:count)
    {
        #pragma omp for schedule(dynamic, chunk) nowait
        for(i = 5; i <= N; i += 2){
            if(isPrime(i) == 1){
                count = count + 1;
            }
        }
    }
    printf("count = %d\n",count);
    return 0;
}
