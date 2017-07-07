/*
Compile:
mpicc -O3 -march=native -std=gnu99 -lm -o HW1_x1054028_advanced HW1_x1054028_advanced.c
mpicc -O3 -march=native -std=gnu99 -lm -o HW1_x1054028_advanced HW1.c
mpicxx -O3 -march=native -std=gnu++03 -lm -o HW1_x1054028 HW1_x1054028.cpp

# large dataset
time mpirun -np 12 ./HW1_x1054028_advanced 376000000 /home/ipl2017/shared/hw1/testcase376000000 out
diff /home/ipl2017/shared/hw1/sorted376000000 out
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <string.h>
#include <ctype.h>
#include <algorithm>
MPI_File in, out;


float* read_file(MPI_File *in, const int head, const int tail){
      MPI_Offset globalstart = head;
      float* chunk;

      {
          MPI_Offset globalend = tail;
          MPI_Offset filesize;

          globalstart *= sizeof(float);
          globalend *= sizeof(float);
          MPI_Offset cur_size = tail - head + 1;

          chunk = new float[(long long)(cur_size) * sizeof(float)];
          MPI_File_seek(*in, globalstart, MPI_SEEK_SET);
          MPI_File_read(*in, chunk, cur_size,
              MPI_FLOAT, MPI_STATUS_IGNORE);
      }

      return chunk;
}

void write_file(MPI_File *out, const int rank, const int head, 
                const int n, float* chunk){
    MPI_Offset globalstart = head;
    globalstart = globalstart * sizeof(float);

    /*int i;
    for (i = 0; i < n; ++i)
        printf("#%d: (%d) %.2f\n", rank, i + 1, chunk[i]);*/
    MPI_File_seek(*out, globalstart, MPI_SEEK_SET);
    MPI_File_write(*out, chunk, 
        n, MPI_FLOAT, MPI_STATUS_IGNORE);
}

int comp(const void * x, const void  * y){
      float f = *((float*)x);
      float s = *((float*)y);
      if (f > s) return 1;
      if (f < s) return -1;
      return 0;
}

void quicksort(float *a, int l, int r){  // sort a[l:r+1]
    int i = l, j = r;
    int temp = a[l];

    while (i < j){
        while (i < j && a[j] >= temp) j--;
        if (i < j)
            a[i++] = a[j];
        while (i < j && a[i] <= temp) i++;
        if (i < j)
            a[j--] = a[i];
    }
    a[i] = temp;

    if(l < i - 1)
        quicksort(a, l, i - 1);
    if(r > i + 1)
        quicksort(a, i + 1, r);
}


int merge_sort(const int na, float *a, const int nb, float *b, float *merged){
    if (na == 0 || nb == 0) return 0;
    if (a[na - 1] <= b[0]) return 0;
    int i = 0, j = 0, cur = 0;
    while (i < na && j < nb){
        if (a[i] <= b[j])
            merged[cur++] = a[i++];
        else
            merged[cur++] = b[j++];
    }
    int res_message = (i == na) && (j == 0);
    if (i == na){
        for (i = 0; i < na; ++i){
            a[i] = merged[i];
        }
        --j;
        for (; j >= 0; --j){
            b[j] = merged[j + na];
        }
    }
    else{
        int pi = i;
        for (i = na - 1, j = nb - 1; i >= pi && j >= 0; --i, --j){
            b[j] = a[i];
        }
        for (; j >= 0; --j){
            b[j] = merged[--cur];
        }
        if (cur < na) merged[cur++] = a[0];
        for (i = na - 1; i >= 0; --i){
            a[i] = merged[--cur];
        }
    }
    return res_message ^ 1;
}

void print(const int na, float *a, const int nb, float *b){
    int i, j;
    printf("[#ELE=%d, %d] ", na, nb);
    for (i = 0; i < na; ++i)
        printf("%.2f ", a[i]);
    printf("##");
    for (j = 0; j < nb; ++j)
        printf(" %.2f", b[j]);
    printf("\n");
}


int main(int argc, char** argv){
    clock_t time_bg, time_ed;
    double time_dur;
    
    time_bg = clock();
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = atoi(argv[1]);

    int n = N / size;
    int mod = N % size;

    int *block_len = new int[(sizeof(int) * size)];
    int *presum_block_len = new int[sizeof(int) * size];

    int i;
    for (i = 0; i < size; ++i){
        block_len[i] = n + (i < mod);
        presum_block_len[i] = block_len[i];
        if (i > 0)
            presum_block_len[i] += presum_block_len[i - 1];
    }

    time_ed = clock();
    time_dur = (double)(time_ed - time_bg) / CLOCKS_PER_SEC;
    printf("Process #%d: Begin time: %.1f secs\n", rank, time_dur);
    time_bg = clock();
    /* read file */
    int ierr = MPI_File_open(MPI_COMM_WORLD,
            argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &in);
    if (ierr) {
          if (rank == 0)
            fprintf(stderr, "%s: Couldn't open file %s\n",
                argv[0], argv[2]);
        MPI_Finalize();
        exit(2);
    }

    int head = presum_block_len[rank] - block_len[rank];
    int tail = presum_block_len[rank] - 1;
    float *chunk = read_file(&in, head, tail);
    MPI_File_close(&in);

    time_ed = clock();
    time_dur = (double)(time_ed - time_bg) / CLOCKS_PER_SEC;
    printf("Process #%d: Read in time: %.1f secs\n", rank, time_dur);
    /* sort invidually */

    time_bg = clock();
    /*qsort(chunk, block_len[rank], sizeof(float), comp);*/
    /*quicksort(chunk, 0, block_len[rank] - 1);*/
    sort(chunk, chunk + block_len[rank]);

    MPI_Barrier(MPI_COMM_WORLD);
    time_ed = clock();
    time_dur = (double)(time_ed - time_bg) / CLOCKS_PER_SEC;
    printf("Process #%d: Qsort time: %.1f secs\n", rank, time_dur);
    /*for (i = 0; i < block_len[rank]; ++i)
        printf("#%d, #%d: %.2f\n", rank, i + 1, chunk[i]);*/

    int res_message[1] = {1};
    int *collection = (int*)malloc(sizeof(int)*size);

    /* even-odd sort */

    time_bg = clock();
    MPI_Status *status;
    MPI_Offset buffer_size = block_len[(rank > 0) ? (rank - 1) : rank];
    float *buffer = new float[(long long)buffer_size * sizeof(float)];
    float *merged = new float[(long long)(buffer_size + block_len[rank]) * sizeof(float)];
    while (1){
        res_message[0] = 0;
        /* odd phrase */

        if (rank % 2 == 1){
              MPI_Recv(buffer, block_len[rank - 1], MPI_FLOAT, rank - 1,
                MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank % 2 == 0 && rank + 1 < size){
            MPI_Send(chunk, block_len[rank], MPI_FLOAT, rank + 1,
                rank + 1, MPI_COMM_WORLD);
        }
        if (rank % 2 == 1)
            res_message[0] = merge_sort(block_len[rank - 1], buffer, block_len[rank], chunk, merged);

        if (rank % 2 == 0 && rank + 1 < size){
            MPI_Recv(chunk, block_len[rank], MPI_FLOAT, rank + 1,
                MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank % 2 == 1){
            MPI_Send(buffer, block_len[rank - 1], MPI_FLOAT, rank - 1,
                rank - 1, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        /* even phrase */

        if (rank % 2 == 0 && rank > 0){
            MPI_Recv(buffer, block_len[rank - 1], MPI_FLOAT, rank - 1,
                MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank % 2 == 1 && rank + 1 < size){
            MPI_Send(chunk, block_len[rank], MPI_FLOAT, rank + 1,
                rank + 1, MPI_COMM_WORLD);
        }
        if (rank % 2 == 0 && rank > 0){
            res_message[0] = merge_sort(block_len[rank - 1], buffer, block_len[rank], chunk, merged);
        }

        if (rank % 2 == 1 && rank + 1 < size){
            MPI_Recv(chunk, block_len[rank], MPI_FLOAT, rank + 1,
                MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank % 2 == 0 && rank > 0){
            MPI_Send(buffer, block_len[rank - 1], MPI_FLOAT, rank - 1,
                rank - 1, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Datatype rtype;
        MPI_Type_contiguous(1, MPI_INT, &rtype);
        MPI_Type_commit(&rtype);
        MPI_Gather(res_message, 1, MPI_INT, collection, 1, rtype,
                0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0){
            int cnt = 0;
            for (i = 0; i < size; ++i)
                cnt += collection[i];
            if (cnt == 0){
                for (i = 0; i < size; ++i)
                    collection[i] = 0;
            }
            else{
                for (i = 0; i < size; ++i)
                    collection[i] = 1;
            }
        }
        MPI_Scatter(collection, 1, rtype, res_message, 1, MPI_INT,
                   0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if (res_message[0] == 0)
            break;
    }

    time_ed = clock();
    time_dur = (double)(time_ed - time_bg) / CLOCKS_PER_SEC;
    printf("Process #%d: OddEven sort time: %.1f secs\n", rank, time_dur);

    /* write file */
    time_bg = clock();
    int oerr = MPI_File_open(MPI_COMM_WORLD,
        argv[3], MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &out);
    if (oerr) {
        if (rank == 0)
            fprintf(stderr, "%s: Couldn't open file %s\n",
                argv[0], argv[3]);
        MPI_Finalize();
        exit(2);
    }

    write_file(&out, rank, head, block_len[rank], chunk);

    time_ed = clock();
    time_dur = (double)(time_ed - time_bg) / CLOCKS_PER_SEC;
    printf("Process #%d: Output time: %.1f secs\n", rank, time_dur);

    time_bg = clock();
    MPI_File_close(&out);
    MPI_Finalize();
    time_ed = clock();
    time_dur = (double)(time_ed - time_bg) / CLOCKS_PER_SEC;
    printf("Process #%d: End time: %.1f secs\n", rank, time_dur);
}
