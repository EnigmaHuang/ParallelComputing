#include <stdio.h>
#include <stdlib.h>

int BLOCK_SIZE;

__global__ void pivot_floyed(int *dist_matrix, int N, int k, int BLOCK_SIZE){
    extern __shared__ int dist_block[];

    int i = threadIdx.x;
    int j = threadIdx.y;
    int x = i + k * BLOCK_SIZE, y = j + k * BLOCK_SIZE;
    dist_block[i * BLOCK_SIZE + j] = dist_matrix[x * N + y];
    __syncthreads();

    int s;
    for (s = 0; s < BLOCK_SIZE; ++s){
        dist_block[i * BLOCK_SIZE + j] = 
            min(dist_block[i * BLOCK_SIZE + j],
                dist_block[i * BLOCK_SIZE + s] + dist_block[s * BLOCK_SIZE + j]);
        __syncthreads();
    }

    dist_matrix[x * N + y] = dist_block[i * BLOCK_SIZE + j];
}

__global__ void pivot_col_floyed(int *dist_matrix, int N, int k, int BLOCK_SIZE){
    extern __shared__ int sdata[];
    int* dist_pivot = sdata;
    int* dist_block = &sdata[BLOCK_SIZE * BLOCK_SIZE];

    int i = threadIdx.x;
    int j = threadIdx.y;

    int r = blockIdx.x;
    r += (r >= k);

    int pivot_x = i + k * BLOCK_SIZE, pivot_y = j + k * BLOCK_SIZE;
    int x = i + r * BLOCK_SIZE;
    dist_block[i * BLOCK_SIZE + j] = dist_matrix[x * N + pivot_y];
    dist_pivot[j * BLOCK_SIZE + i] = dist_matrix[pivot_x * N + pivot_y];
    __syncthreads();

    int s;
    for (s = 0; s < BLOCK_SIZE; ++s){
        dist_block[i * BLOCK_SIZE + j] = 
            min(dist_block[i * BLOCK_SIZE + j],
                dist_block[i * BLOCK_SIZE + s] + dist_pivot[j * BLOCK_SIZE + s]);
        __syncthreads();
    }

    dist_matrix[x * N + pivot_y] = dist_block[i * BLOCK_SIZE + j];
}

__global__ void pivot_row_floyed(int *dist_matrix, int N, int k, int BLOCK_SIZE){
    extern __shared__ int sdata[];
    int* dist_pivot = sdata;
    int* dist_block = &sdata[BLOCK_SIZE * BLOCK_SIZE];

    int i = threadIdx.x;
    int j = threadIdx.y;

    int c = blockIdx.x;
    c += (c >= k);

    int pivot_x = i + k * BLOCK_SIZE, pivot_y = j + k * BLOCK_SIZE;
    int y = j + c * BLOCK_SIZE;
    dist_block[i * BLOCK_SIZE + j] = dist_matrix[pivot_x * N + y];
    dist_pivot[i * BLOCK_SIZE + j] = dist_matrix[pivot_x * N + pivot_y];
    __syncthreads();

    int s;
    for (s = 0; s < BLOCK_SIZE; ++s){
        dist_block[i * BLOCK_SIZE + j] = 
            min(dist_block[i * BLOCK_SIZE + j],
                dist_pivot[i * BLOCK_SIZE + s] + dist_block[s * BLOCK_SIZE + j]);
        __syncthreads();
    }

    dist_matrix[pivot_x * N + y] = dist_block[i * BLOCK_SIZE + j];
}

__global__ void pivot_col_floyed(int *dist_matrix, int N, int k, int BLOCK_SIZE, int *pivot_col_matrix){
    extern __shared__ int sdata[];
    int* dist_pivot = sdata;
    int* dist_block = &sdata[BLOCK_SIZE * BLOCK_SIZE];

    int i = threadIdx.x;
    int j = threadIdx.y;

    int r = blockIdx.x;
    r += (r >= k);

    int pivot_x = i + k * BLOCK_SIZE, pivot_y = j + k * BLOCK_SIZE;
    int x = i + r * BLOCK_SIZE;
    dist_block[i * BLOCK_SIZE + j] = dist_matrix[x * N + pivot_y];
    dist_pivot[j * BLOCK_SIZE + i] = dist_matrix[pivot_x * N + pivot_y];
    __syncthreads();

    int s;
    for (s = 0; s < BLOCK_SIZE; ++s){
        dist_block[i * BLOCK_SIZE + j] = 
            min(dist_block[i * BLOCK_SIZE + j],
                dist_block[i * BLOCK_SIZE + s] + dist_pivot[j * BLOCK_SIZE + s]);
        __syncthreads();
    }

    dist_matrix[x * N + pivot_y] = dist_block[i * BLOCK_SIZE + j];
    pivot_col_matrix[k * BLOCK_SIZE * BLOCK_SIZE + i * BLOCK_SIZE + j] = dist_pivot[i * BLOCK_SIZE + j];
    pivot_col_matrix[r * BLOCK_SIZE * BLOCK_SIZE + i * BLOCK_SIZE + j] = dist_block[i * BLOCK_SIZE + j];
}

__global__ void pivot_row_floyed(int *dist_matrix, int N, int k, int BLOCK_SIZE, int *pivot_row_matrix){
    extern __shared__ int sdata[];
    int* dist_pivot = sdata;
    int* dist_block = &sdata[BLOCK_SIZE * BLOCK_SIZE];

    int i = threadIdx.x;
    int j = threadIdx.y;

    int c = blockIdx.x;
    c += (c >= k);

    int pivot_x = i + k * BLOCK_SIZE, pivot_y = j + k * BLOCK_SIZE;
    int y = j + c * BLOCK_SIZE;
    dist_block[i * BLOCK_SIZE + j] = dist_matrix[pivot_x * N + y];
    dist_pivot[i * BLOCK_SIZE + j] = dist_matrix[pivot_x * N + pivot_y];
    __syncthreads();

    int s;
    for (s = 0; s < BLOCK_SIZE; ++s){
        dist_block[i * BLOCK_SIZE + j] = 
            min(dist_block[i * BLOCK_SIZE + j],
                dist_pivot[i * BLOCK_SIZE + s] + dist_block[s * BLOCK_SIZE + j]);
        __syncthreads();
    }

    dist_matrix[pivot_x * N + y] = dist_block[i * BLOCK_SIZE + j];
    pivot_row_matrix[k * BLOCK_SIZE * BLOCK_SIZE + i * BLOCK_SIZE + j] = dist_pivot[i * BLOCK_SIZE + j];
    pivot_row_matrix[c * BLOCK_SIZE * BLOCK_SIZE + i * BLOCK_SIZE + j] = dist_block[i * BLOCK_SIZE + j];
}

__global__ void res_floyed_full(int *dist_matrix, int N, int k, int BLOCK_SIZE){
    extern __shared__ int sdata[];
    int* dist_pivot_row = sdata;
    int* dist_pivot_col = &sdata[(BLOCK_SIZE) * (BLOCK_SIZE)];

    int i = threadIdx.x;
    int j = threadIdx.y;

    int r = blockIdx.x;
    int c = blockIdx.y;
    r += (r >= k);
    c += (c >= k);

    int pivot_x = i + k * BLOCK_SIZE, pivot_y = j + k * BLOCK_SIZE;
    int x = i + r * BLOCK_SIZE, y = j + c * BLOCK_SIZE;

    dist_pivot_row[j * (BLOCK_SIZE) + i] = dist_matrix[x * N + pivot_y];
    dist_pivot_col[i * (BLOCK_SIZE) + j] = dist_matrix[pivot_x * N + y];
    __syncthreads();

    int s;
    int res = 101 * N, cur;
    for (s = 0; s < BLOCK_SIZE; ++s){
        cur = dist_pivot_row[s * (BLOCK_SIZE) + i] + dist_pivot_col[s * (BLOCK_SIZE) + j];
        if (cur < res) res = cur;
    }

    dist_matrix[x * N + y] = min(dist_matrix[x * N + y], res);
}

__global__ void res_floyed(int *dist_matrix, int N, int k, int BLOCK_SIZE, int divide_line){
    extern __shared__ int sdata[];
    int* dist_pivot_row = sdata;
    int* dist_pivot_col = &sdata[(BLOCK_SIZE) * (BLOCK_SIZE)];

    int i = threadIdx.x;
    int j = threadIdx.y;

    int r = blockIdx.x;
    int c = blockIdx.y;

    if (r < divide_line && c < divide_line)
        return;

    int pivot_x = i + k * BLOCK_SIZE, pivot_y = j + k * BLOCK_SIZE;
    int x = i + r * BLOCK_SIZE, y = j + c * BLOCK_SIZE;

    dist_pivot_row[j * (BLOCK_SIZE) + i] = dist_matrix[x * N + pivot_y];
    dist_pivot_col[i * (BLOCK_SIZE) + j] = dist_matrix[pivot_x * N + y];
    __syncthreads();

    int s;
    int res = 101 * N, cur;
    for (s = 0; s < BLOCK_SIZE; ++s){
        cur = dist_pivot_row[s * (BLOCK_SIZE) + i] + dist_pivot_col[s * (BLOCK_SIZE) + j];
        if (cur < res) res = cur;
    }

    dist_matrix[x * N + y] = min(dist_matrix[x * N + y], res);
}

__global__ void res_floyed(int *dist_matrix, int N, int k, int BLOCK_SIZE, int row_offset, int col_offset){
    extern __shared__ int sdata[];
    int* dist_pivot_row = sdata;
    int* dist_pivot_col = &sdata[(BLOCK_SIZE) * (BLOCK_SIZE)];

    int i = threadIdx.x;
    int j = threadIdx.y;

    int r = blockIdx.x + row_offset;
    int c = blockIdx.y + col_offset;

    int pivot_x = i + k * BLOCK_SIZE, pivot_y = j + k * BLOCK_SIZE;
    int x = i + r * BLOCK_SIZE, y = j + c * BLOCK_SIZE;

    dist_pivot_row[j * (BLOCK_SIZE) + i] = dist_matrix[x * N + pivot_y];
    dist_pivot_col[i * (BLOCK_SIZE) + j] = dist_matrix[pivot_x * N + y];
    __syncthreads();

    int s;
    int res = 101 * N, cur;
    for (s = 0; s < BLOCK_SIZE; ++s){
        cur = dist_pivot_row[s * (BLOCK_SIZE) + i] + dist_pivot_col[s * (BLOCK_SIZE) + j];
        if (cur < res) res = cur;
    }

    dist_matrix[x * N + y] = min(dist_matrix[x * N + y], res);
}

__global__ void res_floyed_slave_ul(int *dist_matrix, int N, int k, int BLOCK_SIZE, int row_offset,
                                    int *pivot_row_matrix, int *pivot_col_matrix){
    extern __shared__ int sdata[];
    int* dist_pivot_row = sdata;
    int* dist_pivot_col = &sdata[(BLOCK_SIZE) * (BLOCK_SIZE)];

    int i = threadIdx.x;
    int j = threadIdx.y;

    int r = blockIdx.x;
    int c = blockIdx.y;

    int x = i + r * BLOCK_SIZE, y = j + c * BLOCK_SIZE;

    dist_pivot_row[j * (BLOCK_SIZE) + i] = pivot_col_matrix[r * (BLOCK_SIZE * BLOCK_SIZE) + i * BLOCK_SIZE + j];
    dist_pivot_col[i * (BLOCK_SIZE) + j] = pivot_row_matrix[c * (BLOCK_SIZE * BLOCK_SIZE) + i * BLOCK_SIZE + j];
    __syncthreads();

    if (r == k){
        dist_matrix[x * N + y] = dist_pivot_col[i * (BLOCK_SIZE) + j];
        return;
    }
    else if (c == k){
        dist_matrix[x * N + y] = dist_pivot_row[j * (BLOCK_SIZE) + i];
        return;
    }

    int s;
    int res = 101 * N, cur;
    for (s = 0; s < BLOCK_SIZE; ++s){
        cur = dist_pivot_row[s * (BLOCK_SIZE) + i] + dist_pivot_col[s * (BLOCK_SIZE) + j];
        if (cur < res) res = cur;
    }

    dist_matrix[x * N + y] = min(res, dist_matrix[x * N + y]);
}

__global__ void res_floyed_slave(int *dist_matrix, int N, int k, int BLOCK_SIZE, int row_offset,
                                 int *pivot_row_matrix, int *pivot_col_matrix){
    extern __shared__ int sdata[];
    int* dist_pivot_row = sdata;
    int* dist_pivot_col = &sdata[(BLOCK_SIZE) * (BLOCK_SIZE)];

    int i = threadIdx.x;
    int j = threadIdx.y;

    int r = blockIdx.x + row_offset;
    int c = blockIdx.y;

    int x = i + r * BLOCK_SIZE, y = j + c * BLOCK_SIZE;

    dist_pivot_row[j * (BLOCK_SIZE) + i] = pivot_col_matrix[r * (BLOCK_SIZE * BLOCK_SIZE) + i * BLOCK_SIZE + j];
    dist_pivot_col[i * (BLOCK_SIZE) + j] = pivot_row_matrix[c * (BLOCK_SIZE * BLOCK_SIZE) + i * BLOCK_SIZE + j];
    __syncthreads();

    if (r == k){
        dist_matrix[x * N + y] = dist_pivot_col[i * (BLOCK_SIZE) + j];
        return;
    }
    else if (c == k){
        dist_matrix[x * N + y] = dist_pivot_row[j * (BLOCK_SIZE) + i];
        return;
    }

    int s;
    int res = 101 * N, cur;
    for (s = 0; s < BLOCK_SIZE; ++s){
        cur = dist_pivot_row[s * (BLOCK_SIZE) + i] + dist_pivot_col[s * (BLOCK_SIZE) + j];
        if (cur < res) res = cur;
    }

    dist_matrix[x * N + y] = res;
}

__global__ void checkmin(int *dist_matrix, int *dist_matrix2, int N, int BLOCK_SIZE){
    int id = (threadIdx.x + (blockIdx.x) * BLOCK_SIZE) * N + (blockIdx.y * BLOCK_SIZE + threadIdx.y);
    int res = dist_matrix2[id];
    if (res < dist_matrix[id])
        dist_matrix[id] = res;
}

void output(int n, int N, int* dist_matrix){
    int i, j;
    printf("=====\n");
    for (i = 0; i < n; ++i)
        for (j = 0; j < n; ++j)
            if (dist_matrix[i * N + j] < N * 101)
                printf((j + 1 < n) ? "%d " : "%d\n", dist_matrix[i * N + j]);
            else
                printf((j + 1 < n) ? "INF " : "INF\n");
}

void output_pivot(int n, int N, int BLOCK_SIZE, int* pivot_matrix){
    int i, j, k;
    printf("-----\n");
    for (k = 0; k < N / BLOCK_SIZE; ++k){
        for (i = 0; i < BLOCK_SIZE; ++i){
            for (j = 0; j < BLOCK_SIZE; ++j)
                printf("%d ", pivot_matrix[k * BLOCK_SIZE * BLOCK_SIZE + i * BLOCK_SIZE + j]);
            printf("\n");
        }
    }
}

#define INPUT_BUF_SIZE 1000000000
#define OUTPUT_BUF_SIZE 1000000000

char input_buf[INPUT_BUF_SIZE], output_buf[OUTPUT_BUF_SIZE];
int input_cur_pt, output_cur_pt;

void bufReRead(){
    printf("new read\n");
    int len = fread(input_buf, 1, INPUT_BUF_SIZE, stdin);
    if (len < INPUT_BUF_SIZE)
        input_buf[len] = '\0';
    input_cur_pt = 0;
}

int getIntFromBuf(){
    char x = ' ';
    while (!(x >= '0' && x <= '9')){
        x = input_buf[input_cur_pt ++];
        if (input_cur_pt == INPUT_BUF_SIZE)
            bufReRead();
    }
    int ret = 0;
    while (x >= '0' && x <= '9'){
        ret = ret * 10 + x - '0';
        x = input_buf[input_cur_pt ++];
        if (input_cur_pt == INPUT_BUF_SIZE)
            bufReRead();
    }
    return ret;
}

void putIntToBuf(int x){
    if (x == 0){
        output_buf[output_cur_pt++] = '0';
        return;
    }
    int len = 0;
    int out[8];
    memset(out, 0, sizeof out);
    for (; ; ){
        int t = x / 10;
        out[++len] = x - t * 10;
        x = t;
        if (x == 0) break;
    }
    for (int i = len; i >= 1; --i)
        output_buf[output_cur_pt++] = out[i] + '0';
}


int main(int argc, char** argv){
    char *input_filename = argv[1];
    char *output_filename = argv[2];

    BLOCK_SIZE = atoi(argv[3]);
    BLOCK_SIZE = min(BLOCK_SIZE, 32);

    /* input & output device */
    input_cur_pt = 0;
    output_cur_pt = 0;
    freopen(input_filename, "r", stdin);
    int len = fread(input_buf, 1, INPUT_BUF_SIZE, stdin);
    if (len < INPUT_BUF_SIZE)
        input_buf[len] = '\0';

    /*
    FOR CUDA
    if (BLOCK_SIZE < 32 && BLOCK_SIZE >= 24) BLOCK_SIZE = 24;
    if (BLOCK_SIZE < 24 && BLOCK_SIZE >= 16) BLOCK_SIZE = 16;
    if (BLOCK_SIZE < 16 && BLOCK_SIZE >= 8) BLOCK_SIZE = 8;
    if (BLOCK_SIZE < 8) BLOCK_SIZE = 8;
    */

    int i, j;
    int n, m;

    /*scanf("%d%d", &n, &m);*/
    n = getIntFromBuf();
    m = getIntFromBuf();

    /* Padding */
    int num_blocks = n / BLOCK_SIZE;
    if (num_blocks * BLOCK_SIZE < n)
        num_blocks ++;
    int N = num_blocks * BLOCK_SIZE;
    int* dist_matrix = (int*)malloc(sizeof(int) * N * N);

    /* read in data */
    for (i = 0; i < N * N; ++i)
        dist_matrix[i] = N * 101;

    for (i = 0; i < N; ++i)
        dist_matrix[i * N + i] = 0;

    for (i = 0; i < m; ++i){
        int x, y, w;
        /*scanf("%d%d%d", &x, &y, &w);*/
        x = getIntFromBuf();
        y = getIntFromBuf();
        w = getIntFromBuf();
        x--;
        y--;
        if (dist_matrix[x * N + y] > w)
            dist_matrix[x * N + y] = w;
    }

    int* d_dist_matrix;
    int* d_pivot_row;
    int* d_pivot_col;
    int* vd_dist_matrix;
    int* vd_pivot_row;
    int* vd_pivot_col;
    int* d_bak_matrix;
    int size = sizeof(int) * N * N;
    int pivot_line_size = sizeof(int) * N * BLOCK_SIZE;
    int *pivot_row = (int*)malloc(pivot_line_size);
    int *pivot_col = (int*)malloc(pivot_line_size);

    /* block ASPA */
    cudaStream_t stream[2];
    cudaEvent_t fin, fin0[num_blocks];

    cudaSetDevice(0);
    cudaStreamCreate(&stream[0]);
    for (i = 0; i < num_blocks; ++i)
        cudaEventCreate(&fin0[i]);
    cudaMalloc((void**)&d_dist_matrix, size);
    cudaMalloc((void**)&d_pivot_row, pivot_line_size);
    cudaMalloc((void**)&d_pivot_col, pivot_line_size);
    cudaMalloc((void**)&d_bak_matrix, size);

    cudaMemcpy(d_dist_matrix, dist_matrix, size, cudaMemcpyHostToDevice);
    cudaSetDevice(1);
    cudaStreamCreate(&stream[1]);
    cudaEventCreate(&fin);
    cudaMalloc((void**)&vd_dist_matrix, size);
    cudaMalloc((void**)&vd_pivot_row, pivot_line_size);
    cudaMalloc((void**)&vd_pivot_col, pivot_line_size);
    cudaMemcpy(vd_dist_matrix, dist_matrix, size, cudaMemcpyHostToDevice);
    cudaDeviceEnablePeerAccess(0, 0);

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);

    if (num_blocks > 4){
        int divide_line = num_blocks / sqrt(2);
        for (i = 0; i < num_blocks; ++i){
            /* phrase #1: self dependent blocks */
            dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

            int master_task = (num_blocks - 1) * 0.70;
            int num_row_blocks = master_task + (i <= master_task);
            int slave_task = num_blocks - num_row_blocks;

            dim3 blockPerGrid_master(num_blocks - 1, num_blocks - 1);
            cudaSetDevice(0);
            pivot_floyed<<<1, threadsPerBlock,
                sizeof(int) * (BLOCK_SIZE) * (BLOCK_SIZE), stream[0]>>>(d_dist_matrix, N, i, BLOCK_SIZE);
            /* phrase #2: pivot row & col blocks */
            pivot_row_floyed<<<num_blocks - 1, threadsPerBlock,
                sizeof(int) * (BLOCK_SIZE) * (BLOCK_SIZE) * 2, stream[0]>>>(d_dist_matrix, N, i, BLOCK_SIZE, d_pivot_row);
            pivot_col_floyed<<<num_blocks - 1, threadsPerBlock,
                sizeof(int) * (BLOCK_SIZE) * (BLOCK_SIZE) * 2, stream[0]>>>(d_dist_matrix, N, i, BLOCK_SIZE, d_pivot_col);

            cudaMemcpyPeerAsync(vd_pivot_col, 1, d_pivot_col, 0, pivot_line_size, stream[0]);
            cudaMemcpyPeerAsync(vd_pivot_row, 1, d_pivot_row, 0, pivot_line_size, stream[0]);

            cudaEventRecord(fin0[i], stream[0]);
            /* phrase #3: other blocks */
            if (i + 1 < num_blocks){
                res_floyed<<<dim3(num_blocks - i - 1, num_blocks - i - 1), threadsPerBlock,
                    sizeof(int) * (BLOCK_SIZE) * (BLOCK_SIZE) * 2, stream[0]>>>(d_dist_matrix, N, i, BLOCK_SIZE, i + 1, i + 1);
                if (i > 0){
                    res_floyed<<<dim3(i, num_blocks - i - 1), threadsPerBlock,
                        sizeof(int) * (BLOCK_SIZE) * (BLOCK_SIZE) * 2, stream[0]>>>(d_dist_matrix, N, i, BLOCK_SIZE, 0, i + 1);
                    res_floyed<<<dim3(num_blocks - i - 1, i), threadsPerBlock,
                        sizeof(int) * (BLOCK_SIZE) * (BLOCK_SIZE) * 2, stream[0]>>>(d_dist_matrix, N, i, BLOCK_SIZE, i + 1, 0);
                }
            }
            if (i > divide_line && i > 0){
                res_floyed<<<dim3(i, i), threadsPerBlock,
                    sizeof(int) * (BLOCK_SIZE) * (BLOCK_SIZE) * 2, stream[0]>>>(d_dist_matrix, N, i, BLOCK_SIZE, divide_line);
            }

            cudaSetDevice(1);
            cudaStreamWaitEvent(stream[1], fin0[i], 0);
            if (i > 0){
                dim3 blockPerGrid_slave(min(i, divide_line), min(i, divide_line));
                res_floyed_slave_ul<<<blockPerGrid_slave, threadsPerBlock,
                    sizeof(int) * (BLOCK_SIZE) * (BLOCK_SIZE) * 2, stream[1]>>>(vd_dist_matrix, N, i, BLOCK_SIZE, num_row_blocks, 
                                                                                vd_pivot_row, vd_pivot_col);
            }
        }
        cudaSetDevice(1);
        cudaMemcpyPeerAsync(d_bak_matrix, 0, vd_dist_matrix, 1, size, stream[1]);
        cudaEventRecord(fin, stream[1]);
        cudaSetDevice(0);
        dim3 blockPerGrid_slave(num_blocks, num_blocks);
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        cudaStreamWaitEvent(stream[0], fin, 0);
        checkmin<<<blockPerGrid_slave, threadsPerBlock, 0, stream[0]>>>(d_dist_matrix, d_bak_matrix, N, BLOCK_SIZE);
    }
    else
        for (i = 0; i < num_blocks; ++i){
            /* phrase #1: self dependent blocks */
            dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 blockPerGrid(num_blocks - 1, num_blocks - 1);
            pivot_floyed<<<1, threadsPerBlock, 
                sizeof(int) * (BLOCK_SIZE) * (BLOCK_SIZE)>>>(d_dist_matrix, N, i, BLOCK_SIZE);
            if (num_blocks > 1){
                /* phrase #2: pivot row & col blocks */
                pivot_row_floyed<<<num_blocks - 1, threadsPerBlock,
                    sizeof(int) * (BLOCK_SIZE) * (BLOCK_SIZE) * 2>>>(d_dist_matrix, N, i, BLOCK_SIZE);
                pivot_col_floyed<<<num_blocks - 1, threadsPerBlock,
                    sizeof(int) * (BLOCK_SIZE) * (BLOCK_SIZE) * 2>>>(d_dist_matrix, N, i, BLOCK_SIZE);
                /* phrase #3: other blocks */
                res_floyed_full<<<blockPerGrid, threadsPerBlock,
                    sizeof(int) * (BLOCK_SIZE) * (BLOCK_SIZE) * 2>>>(d_dist_matrix, N, i, BLOCK_SIZE);
            }
        }

    cudaMemcpy(dist_matrix, d_dist_matrix, size, cudaMemcpyDeviceToHost);

    freopen(output_filename, "w", stdout);
    for (i = 0; i < n; ++i){
        for (j = 0; j < n; ++j){
            if (dist_matrix[i * N + j] < N * 101)
                putIntToBuf(dist_matrix[i * N + j]);
            else{
                output_buf[output_cur_pt++] = 'I';
                output_buf[output_cur_pt++] = 'N';
                output_buf[output_cur_pt++] = 'F';
            }
            output_buf[output_cur_pt++] = ' ';
        }
        output_buf[output_cur_pt++] = '\n';
    }
    fwrite(output_buf, 1, output_cur_pt, stdout);
}
