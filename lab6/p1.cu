#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void reduceSum(int *g_idata, int *g_odata){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    unsigned int s;
    for (s = 1; s < blockDim.x; s *= 2){
        if (tid % (2 * s) == 0)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

int main(int argc, char** argv){
    srand(2634);
    int N = atoi(argv[1]);
    char* out = argv[2];

    cudaEvent_t start, stop;
    float dur_time;

    size_t size = N * sizeof(int);
    int* h_in = (int*)malloc(size);
    int* h_out = (int*)malloc(size);

    int i;
    for (i = 0; i < N; ++i){
        h_in[i] = rand() % 41;
    }

    int* d_in;
    int* d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStream_t stream0;
    cudaStreamCreate(&stream0);

    cudaEventRecord(start, stream0);
    int threadsPerBlock = 256;
    for (i = N; i >= threadsPerBlock; i = i / threadsPerBlock){
        int blocksPerGrid = i / threadsPerBlock;
        printf("%d %d\n", blocksPerGrid, threadsPerBlock);
        reduceSum<<<blocksPerGrid, threadsPerBlock, sizeof(int) * threadsPerBlock, stream0>>>(d_in, d_in);
    }
    cudaEventRecord(stop, stream0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dur_time, start, stop);
    fprintf(stderr, "%.3f\n", dur_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_out, d_in, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    printf("%d\n", h_out[0]);

    free(h_in);
    free(h_out);
    return 0;
}