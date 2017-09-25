#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <pthread.h>
#include <time.h>
#include <iostream>
using namespace std;

int width = 41, height;
int cur_x, cur_y;
int num_lane;
double prop[5] = {0.1, 1.0, 0.7, 0.4, 0.0};
int TIME_BASE, TIME_RAND;
int board[55][42];
int game_go_on;
int time_left, tot_time;
pthread_mutex_t board_print_mutex, board_restart_mutex;
pthread_cond_t board_print_cond, board_restart_cond;

char getch(){
    char buf = 0;
    struct termios old = {0};
    if (tcgetattr(0, &old) < 0)
        perror("tcsetattr()");
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;
    if (tcsetattr(0, TCSANOW, &old) < 0)
        perror("tcsetattr ICANON");
    if (read(0, &buf, 1) < 0)
        perror("read()");
    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    if (tcsetattr(0, TCSADRAIN, &old) < 0)
        perror ("tcsetattr ~ICANON");
    return (buf);
}

int lane_move(int id, int dir, int cur_tail){
    int ct = ((double) rand() / RAND_MAX) < prop[cur_tail];
    if (dir == 0){
        /* -> */
        for (int i = width - 1; i >= 1; --i)
            board[id + 1][i] = board[id + 1][i - 1];
        board[id + 1][0] = ct;
    }
    else{
        /* <- */
        for (int i = 0; i < width - 1; ++i)
            board[id + 1][i] = board[id + 1][i + 1];
        board[id + 1][width - 1] = ct;
    }
    if (ct == 1) cur_tail ++; else cur_tail = 0;
    return cur_tail;
}

void *lane_sim(void* arg){
    int dir = rand() % 2;
    int id = ((int*)arg)[0];
    int period = ((int*)arg)[1];
    for (int i = 0; i < width; ++i)
        board[id + 1][i] = 0;
    int cur_tail = 0;
    for (int i = 0; i < width; ++i)
        cur_tail = lane_move(id, dir, cur_tail);
    while (1){
        cur_tail = lane_move(id, dir, cur_tail);
        pthread_mutex_lock(&board_print_mutex);
        pthread_cond_signal(&board_print_cond);
        pthread_mutex_unlock(&board_print_mutex);
        usleep(period);
    }
    pthread_exit(NULL);
}

void *board_print(void* arg){
    while (1){
        pthread_mutex_lock(&board_print_mutex);
        pthread_cond_wait(&board_print_cond, &board_print_mutex);
        pthread_mutex_unlock(&board_print_mutex);
        system("clear");

        if (cur_x == 0){
            game_go_on = 2;
        }
        if (board[cur_x][cur_y] == 1){
            game_go_on = 0;
        }
        if (time_left == 0){
            game_go_on = -1;
        }

        printf("+");
        for (int i = 0; i < width; ++i) printf("=");
        printf("+\n");
        printf("|                 FROGGER                 |\n");
        printf("+");
        for (int i = 0; i < width; ++i) printf("=");
        printf("+\n");
        for (int i = 0; i < num_lane + 2; ++i){
            printf("|");
            for (int j = 0; j < width; ++j){
                if (i == cur_x && j == cur_y)
                    printf("X");
                else
                if (board[i][j] == 1) printf("#"); else printf(" ");
            }
            printf("|\n");
        }
        printf("+");
        for (int i = 0; i < width; ++i) printf("-");
        printf("+\n");
        if (game_go_on == 1){
            printf("|          GO ON: Time Left %03d s         |\n", time_left);
        }
        if (game_go_on == 0){
            printf("|              Failed: OUCH!              |\n");
        }
        if (game_go_on == -1){
            printf("|          Failed: No Time Left!          |\n");
        }
        if (game_go_on == 2){
            printf("|                 Success                 |\n");
        }

        printf("+");
        for (int i = 0; i < width; ++i) printf("-");
        printf("+\n");

        if (game_go_on != 1){
            printf("|      Press 'R' to Restart Game ...      |\n");
            printf("+");
            for (int i = 0; i < width; ++i) printf("-");
            printf("+\n");
            pthread_mutex_lock(&board_restart_mutex);
            pthread_cond_wait(&board_restart_cond, &board_restart_mutex);
            pthread_mutex_unlock(&board_restart_mutex);
        }
    }
    pthread_exit(NULL);
}

void *cursor_manager(void *arg){
    while (1){
        char ch = getch();
        if (ch == 'w' || ch == 'W'){
            if (cur_x > 0) cur_x --;
        }
        if (ch == 's' || ch == 'S'){
            if (cur_x < height - 1) cur_x++;
        }
        if (ch == 'a' || ch == 'A'){
            if (cur_y > 0) cur_y --;
        }
        if (ch == 'd' || ch == 'D'){
            if (cur_y < width - 1) cur_y ++;
        }
        if ((ch == 'r' || ch == 'R') && game_go_on != 1){
            game_go_on = 1;
            cur_x = height - 1;
            cur_y = width / 2 + 1;
            time_left = tot_time;
            pthread_mutex_lock(&board_restart_mutex);
            pthread_cond_signal(&board_restart_cond);
            pthread_mutex_unlock(&board_restart_mutex);
        }
        pthread_mutex_lock(&board_print_mutex);
        pthread_cond_signal(&board_print_cond);
        pthread_mutex_unlock(&board_print_mutex);
    }
    pthread_exit(NULL);
}

void *time_manager(void *arg){
    while (time_left > 0){
        usleep(1000000);
        time_left --;
        pthread_mutex_lock(&board_print_mutex);
        pthread_cond_signal(&board_print_cond);
        pthread_mutex_unlock(&board_print_mutex);        
    }
    pthread_exit(NULL);
}

int main(int argc, char* argv[]){
    srand(time(0));
    num_lane = atoi(argv[1]);
    time_left = atoi(argv[2]);
    int difficulty = atoi(argv[3]);
    if (difficulty == 0){
        /* easy */
        prop[0] = 0.08;
        TIME_RAND = 1500;
        TIME_BASE = 500;
    }
    if (difficulty == 1){
        /* medium */
        prop[0] = 0.10;
        TIME_RAND = 800;
        TIME_BASE = 200;
    }
    if (difficulty == 2){
        /* hard */
        prop[0] = 0.15;
        TIME_RAND = 450;
        TIME_BASE = 50;
    }
    tot_time = time_left;
    game_go_on = 1;
    height = num_lane + 2;
    cur_x = height - 1;
    cur_y = width / 2 + 1;
    pthread_cond_init(&board_print_cond, NULL);
    pthread_cond_init(&board_restart_cond, NULL);
    pthread_mutex_init(&board_print_mutex, NULL);
    pthread_mutex_init(&board_restart_mutex, NULL);

    pthread_t lane_thread[num_lane];
    pthread_t board_print_thread;
    pthread_t cursor_manager_thread;
    pthread_t time_manager_thread;

    pthread_create(&board_print_thread, NULL, board_print, NULL);

    for (int i = 0; i < num_lane; ++i){
        int *buf = new int[2];
        buf[0] = i;
        buf[1] = rand() % TIME_RAND + TIME_BASE;
        buf[1] *= 1000;
        pthread_create(&lane_thread[i], NULL, lane_sim, buf);
    }

    pthread_create(&cursor_manager_thread, NULL, cursor_manager, NULL);
    pthread_create(&time_manager_thread, NULL, time_manager, NULL);

    pthread_join(cursor_manager_thread, NULL);
}