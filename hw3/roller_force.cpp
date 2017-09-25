#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <fstream>
using namespace std;

int cur_time[11];

int cmp(int x, int y){
    if (cur_time[x] != cur_time[y])
        return cur_time[x] < cur_time[y];
    else{
        return x < y;
    }
}

int main(int argc, char** argv){
    int n = atoi(argv[1]);
    int C = atoi(argv[2]);
    int T = atoi(argv[3]);
    int N = atoi(argv[4]);
    C = min(C, n);
    char* output_filename = argv[5];
    freopen(output_filename, "w", stdout);
    printf("%d %d %d %d\n", n, C, T, N);
    for (int i = 1; i <= n; ++i){
        cur_time[i] = i;
        printf("Passenger %d wanders around the park.\n", i);
    }
    int queue[10];
    for (int i = 0; i < n; ++i) queue[i] = i + 1;

    int arriv_time = -1;
    int i, t;
    int newround[10];
    int nround = 0;
    for (i = 0, t = 0; i < N; ++t){
        nround = 0;
        for (int id = 1; id <= n; ++id)
            if (cur_time[id] == t)
                printf("Passenger %d returns for another ride at %d millisec.\n", id, cur_time[id]);
        sort(queue, queue + n, cmp);
        if (t == arriv_time){
            printf("Car arrives at %d millisec. Passenger", arriv_time);
            for (int j = 0; j < C; ++j){
                printf(" %d", queue[j]);
            }
            printf(" get off the car.\n");
            ++i;
            if (i == N) break;
            sort(queue, queue + C);
            for (int j = 0; j < C; ++j){
                newround[j] = queue[j];
                cur_time[queue[j]] = arriv_time + queue[j];
            }
            nround = 1;
        }
        sort(queue, queue + n, cmp);
        int depart_time = cur_time[queue[C - 1]];
        if (nround == 1){
            depart_time = max(depart_time, arriv_time);
        }
        if (depart_time == t){
            printf("Car departures at %d millisec. Passenger", depart_time);
            for (int j = 0; j < C; ++j)
                printf(" %d", queue[j]);
            printf(" are in the car.\n");
            arriv_time = depart_time + T;
        }
        if (nround == 1){
            for (int j = 0; j < C; ++j)
                printf("Passenger %d wanders around the park.\n", newround[j]);
        }
    }
    sort(queue, queue + n, cmp);
    for (int i = 0; i < n; ++i){
        if (cur_time[queue[i]] > t)
            printf("Passenger %d returns for another ride at %d millisec.\n", queue[i], cur_time[queue[i]]);
    }
}