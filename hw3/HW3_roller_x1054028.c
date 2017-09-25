#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>

int num_passengers, capacity, T, num_sim;
int queue[11], cur_time[11], roller_passenger[11];
int count;
pthread_mutex_t mutex_time[11];
pthread_mutex_t mutex_queue;
pthread_mutex_t mutex_release[11];
pthread_cond_t launch_cond, release_cond[11];
pthread_mutex_t mutex_output;
int *psg_event_time_pt[11];
int *psg_event_type_pt[11];
int psg_event_cur[11];
int *roller_event_time_pt;
int **roller_event_board;

void *sim_passenger(void* arg){
    int id = *(int*)arg;
    while (1){
        /* wander */
        pthread_mutex_lock(&mutex_output);
        /*printf("Passenger %d wanders around the park.\n", id);*/
        psg_event_type_pt[id][psg_event_cur[id]] = 3;
        psg_event_time_pt[id][psg_event_cur[id]] = cur_time[id];
        psg_event_cur[id]++;
        pthread_mutex_unlock(&mutex_output);

        usleep(id * 1000);

        pthread_mutex_lock(&mutex_time[id]);
        cur_time[id] += id;
        pthread_mutex_unlock(&mutex_time[id]);

        pthread_mutex_lock(&mutex_output);
        /*printf("Passenger %d returns for another ride at %d millisec.\n", id, cur_time[id]);*/
        psg_event_type_pt[id][psg_event_cur[id]] = 0;
        psg_event_time_pt[id][psg_event_cur[id]] = cur_time[id];
        psg_event_cur[id]++;
        pthread_mutex_unlock(&mutex_output);

        /* get in the queue */
        pthread_mutex_lock(&mutex_queue);
        queue[count++] = id;
        /* modified */
        if (count >= num_passengers)
            pthread_cond_signal(&launch_cond);
        pthread_mutex_unlock(&mutex_queue);

        pthread_mutex_lock(&mutex_release[id]);
        pthread_cond_wait(&release_cond[id], &mutex_release[id]);
        pthread_mutex_unlock(&mutex_release[id]);
    }
}

int comp(const void* a, const void* b){
    int x = *(int*)a, y = *(int*)b;
    if (cur_time[x] < cur_time[y] || 
        (cur_time[x] == cur_time[y] && x < y))
        return -1;
    if (cur_time[x] > cur_time[y] || 
        (cur_time[x] == cur_time[y] && x > y))
        return 1;
    return 0;
}

void *sim_roller_coaster(){
    int round = 0;
    int lst_arriv_time = 0;
    for (; round < num_sim; ++round){
        pthread_mutex_lock(&mutex_queue);
        if (count < num_passengers)
            pthread_cond_wait(&launch_cond, &mutex_queue);

        int i;
        qsort(queue, count, sizeof(int), comp);

        int depart_time = 0;
        for (i = 0; i < capacity; ++i)
            if (cur_time[queue[i]] > depart_time)
                depart_time = cur_time[queue[i]];
        if (lst_arriv_time > depart_time)
            depart_time = lst_arriv_time;
        int arriv_time = depart_time + T;
        lst_arriv_time = arriv_time;

        for (i = 0; i < capacity; ++i)
            roller_passenger[i] = queue[i];
        for (i = 0; i < count - capacity; ++i)
            queue[i] = queue[i + capacity];
        count -= capacity;
        pthread_mutex_unlock(&mutex_queue);
        
        pthread_mutex_lock(&mutex_output);

        roller_event_time_pt[round * 2] = depart_time;
        for (i = 0; i < capacity; ++i)
            roller_event_board[round * 2][i] = roller_passenger[i];

        /*printf("Car departures at %d millisec. Passenger", depart_time);
        for (i = 0; i < capacity; ++i)
            printf(" %d", roller_passenger[i]);
        printf(" are in the car.\n");*/
        pthread_mutex_unlock(&mutex_output);

        usleep(T * 1000);

        pthread_mutex_lock(&mutex_output);

        roller_event_time_pt[round * 2 + 1] = arriv_time;
        for (i = 0; i < capacity; ++i)
            roller_event_board[round * 2 + 1][i] = roller_passenger[i];

        /*printf("Car arrives at %d millisec. Passenger", arriv_time);
        for (i = 0; i < capacity; ++i)
            printf(" %d", roller_passenger[i]);
        printf(" get off the car.\n");*/
        pthread_mutex_unlock(&mutex_output);

        if (round + 1 == num_sim)
            break;

        for (i = 0; i < capacity; ++i){
            pthread_mutex_lock(&mutex_time[roller_passenger[i]]);
            cur_time[roller_passenger[i]] = arriv_time;
            pthread_mutex_unlock(&mutex_time[roller_passenger[i]]);
        }

        for (i = 0; i < capacity; ++i){
            pthread_mutex_lock(&mutex_release[roller_passenger[i]]);
            pthread_cond_signal(&release_cond[roller_passenger[i]]);
            pthread_mutex_unlock(&mutex_release[roller_passenger[i]]);
        }
    }
    pthread_exit(NULL);
}

void roller_output(int idx){
    int i;
    if (idx % 2 == 0){
        printf("Car departures at %d millisec. Passenger", roller_event_time_pt[idx]);
        for (i = 0; i < capacity; ++i)
            printf(" %d", roller_event_board[idx][i]);
        printf(" are in the car.\n");
    }
    else{
        printf("Car arrives at %d millisec. Passenger", roller_event_time_pt[idx]);
        for (i = 0; i < capacity; ++i)
            printf(" %d", roller_event_board[idx][i]);
        printf(" get off the car.\n");
    }
}

void passenger_output(int id, int idx){
    if (psg_event_type_pt[id][idx] == 0)
        printf("Passenger %d returns for another ride at %d millisec.\n", id, psg_event_time_pt[id][idx]);
    else
        printf("Passenger %d wanders around the park.\n", id);
}

int main(int argc, char** argv){
    num_passengers = atoi(argv[1]);
    capacity = atoi(argv[2]);
    T = atoi(argv[3]);
    num_sim = atoi(argv[4]);
    char* output_filename = argv[5];
    freopen(output_filename, "w", stdout);

    if (capacity > num_passengers)
        capacity = num_passengers;

    pthread_t passanger[num_passengers];
    pthread_t roller_coaster;

    count = 0;
    int i;
    for (i = 1; i <= num_passengers; ++i){
        cur_time[i] = 0;
        psg_event_cur[i] = 0;
        psg_event_type_pt[i] = (int*)malloc(sizeof(int) * num_sim * 2);
        psg_event_time_pt[i] = (int*)malloc(sizeof(int) * num_sim * 2); 
    }
    roller_event_time_pt = (int*)malloc(sizeof(int) * num_sim * 2);
    roller_event_board = (int* *)malloc(sizeof(int*) * num_sim * 2);
    for (i = 0; i < num_sim * 2; ++i)
        roller_event_board[i] = (int*)malloc(sizeof(int) * capacity);

    for (i = 1; i <= num_passengers; ++i){
        pthread_mutex_init(&mutex_time[i], NULL);
        pthread_mutex_init(&mutex_release[i], NULL);
        pthread_cond_init(&release_cond[i], NULL);
    }

    pthread_cond_init(&launch_cond, NULL);
    pthread_mutex_init(&mutex_output, NULL);
    pthread_mutex_init(&mutex_queue, NULL);

    pthread_create(&roller_coaster, NULL, sim_roller_coaster, NULL);
    for (i = 1; i <= num_passengers; ++i){
        int* buf = (int*)malloc(sizeof(int) * 1);
        buf[0] = i;
        pthread_create(&passanger[i - 1], NULL, sim_passenger, buf);
    }

    pthread_join(roller_coaster, NULL);
    usleep(2 * num_passengers * 1000);

    /* output */
    printf("%d %d %d %d\n", num_passengers, capacity, T, num_sim);
    int pass_head[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int roller_head = 0;
    while (1){
        int id = -1;
        for (i = 0; i <= num_passengers; ++i){
            if (id == -1 && pass_head[i] < psg_event_cur[i])
                id = i;
            if (id == -1) continue;
            if (pass_head[i] == psg_event_cur[i]) continue;
            if (psg_event_time_pt[i][pass_head[i]] < psg_event_time_pt[id][pass_head[id]] || 
                  (psg_event_time_pt[i][pass_head[i]] == psg_event_time_pt[id][pass_head[id]]
                    && psg_event_type_pt[i][pass_head[i]] < psg_event_type_pt[id][pass_head[id]]))
                id = i;
        }
        if (id == -1 && roller_head == 2 * num_sim)
            break;
        if (id == -1){
            roller_output(roller_head);
            roller_head ++;
            continue;
        }
        if (roller_head == 2 * num_sim){
            passenger_output(id, pass_head[id]);
            pass_head[id] ++;
            continue;
        }
        if (psg_event_time_pt[id][pass_head[id]] < roller_event_time_pt[roller_head] || 
              (psg_event_time_pt[id][pass_head[id]] == roller_event_time_pt[roller_head] && 
                psg_event_type_pt[id][pass_head[id]] == 0)){
            passenger_output(id, pass_head[id]);
            pass_head[id] ++;
        }
        else{
            roller_output(roller_head);
            roller_head ++;
        }
    }
}