#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void VecAdd(float* A, float* B, float* C, int N){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N)
        C[i] = A[i] * A[i] + B[i] * B[i];
}

int main(int argc, char** argv){
    srand(2634);
    int N = atoi(argv[1]);
    char* out = argv[2];

    cudaEvent_t start, stop;
    float dur_time;

    size_t size = N * sizeof(float);
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    int i;
    for (i = 0; i < N; ++i){
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float* d_A;
    cudaMalloc((void**)&d_A, size);
    float* d_B;
    cudaMalloc((void**)&d_B, size);
    float* d_C;
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dur_time, start, stop);
    fprintf(stderr, "%.3f\n", dur_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    freopen(out, "w", stdout);
    for (i = 0; i < N; ++i)
        printf("%.7f\n", h_C[i]);

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}