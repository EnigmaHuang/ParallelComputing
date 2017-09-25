#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void VecAdd(float* A, float* B, float* C, int N){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main(int argc, char** argv){
    srand(2634);
    int N = atoi(argv[1]);
    char* out = argv[2];

    cudaEvent_t start, stop, fin;
    float dur_time;

    size_t size = N * sizeof(float);
    float* h_A;
    cudaMallocHost((void**)&h_A, size);
    float* h_B;
    cudaMallocHost((void**)&h_B, size);
    float* h_C;
    cudaMallocHost((void**)&h_C, size);
    float* h_D;
    cudaMallocHost((void**)&h_D, size);
    float* h_F;
    cudaMallocHost((void**)&h_F, size);

    int i;
    for (i = 0; i < N; ++i){
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
        h_D[i] = (float)rand() / RAND_MAX;
    }

    float* d_A;
    cudaMalloc((void**)&d_A, size);
    float* d_B;
    cudaMalloc((void**)&d_B, size);
    float* d_C;
    cudaMalloc((void**)&d_C, size);
    float* d_D;
    cudaMalloc((void**)&d_D, size);
    float* d_F;
    cudaMalloc((void**)&d_F, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&fin);

    cudaEventRecord(start, 0);
    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream[0]);
    VecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream[0]>>>(d_A, d_B, d_C, N);
    cudaEventRecord(fin, stream[0]);
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream[0]);

    cudaMemcpyAsync(d_D, h_D, size, cudaMemcpyHostToDevice, stream[1]);
    cudaStreamWaitEvent(stream[1], fin, 0);
    VecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream[1]>>>(d_C, d_D, d_F, N);
    cudaMemcpyAsync(h_F, d_F, size, cudaMemcpyDeviceToHost, stream[1]);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dur_time, start, stop);
    fprintf(stderr, "%.3f\n", dur_time);
    cudaEventDestroy(fin);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_F);

    freopen(out, "w", stdout);
    for (i = 0; i < N; ++i)
        printf("%.5f %.5f\n", h_C[i], h_F[i]);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_D);
    cudaFreeHost(h_F);

    return 0;
}