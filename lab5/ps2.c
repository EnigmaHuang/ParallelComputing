#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv){
    srand(2634);
    int N = atoi(argv[1]);
    char* out = argv[2];
    int size = N * sizeof(float);
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    float* h_D = (float*)malloc(size);
    float* h_F = (float*)malloc(size);

    int i;
    for (i = 0; i < N; ++i){
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
        h_D[i] = (float)rand() / RAND_MAX;
    }
    clock_t time_bg, time_ed;
    time_bg = clock();
    for (i = 0; i < N; ++i){
        h_C[i] = h_A[i] + h_B[i];
        h_F[i] = h_D[i] + h_C[i];
    }
    time_ed = clock();
    float time_dur = (double)(time_ed - time_bg) / CLOCKS_PER_SEC;
    printf("%.3f secs\n", time_dur);

    freopen(out, "w", stdout);
    for (i = 0; i < N; ++i)
        printf("%.5f %.5f\n", h_C[i], h_F[i]);
    return 0;
}