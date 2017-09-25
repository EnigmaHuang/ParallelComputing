#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <string.h>
#include <ctype.h>

void *mc_sim(void *arg){
    int *pt = (int*)arg;
    int n = *(pt + 0);
    int id = *(pt + 1);
    printf("Thread %d: Start working, %d simulations.\n", id, n);
    int cnt = 0;
    for (int i = 0; ; ++i){
        double x = (double) rand() / RAND_MAX;
        double y = (double) rand() / RAND_MAX;
        cnt += (x * x + y * y) <= 1.0;
        printf("Thread %d: %d, %d, %d\n", id, i, n, cnt);
        if (i == n - 1) break;
    }
    double *res = (double*)malloc(sizeof(double));
    res[0] = cnt;
    printf("Thread %d: Result = %.7f\n", id, res[0] / n * 4);
    pthread_exit((void*)res);
}

int main(int argc, char** argv){
    int num_sim = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    int n = num_sim / num_threads;
    int i;
    pthread_t tid[num_threads];
    void* results[num_threads];
    for (i = 0; i < num_threads; ++i){
        int* buf = (int*)malloc(sizeof(int) * 2);
        buf[0] = n + (i < (num_sim % num_threads));
        buf[1] = i;
        pthread_create(&tid[i], NULL, mc_sim, buf);
    }
    for (int i = 0; i < num_threads; ++i)
        pthread_join(tid[i], &results[i]);
    double sum = 0;
    for (int i = 0; i < num_threads; ++i){
        sum = sum + *(double*)results[i];
        printf("%.2f\n", *(double*)results[i]);
    }
    double ans = sum / num_sim * 4;
    printf("Answer = %.7f\n", ans);
}
