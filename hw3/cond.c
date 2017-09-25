#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
int arr[10];
pthread_cond_t cond;
pthread_mutex_t mutex;

void *fill() {
  int i=0;
  printf("Enter 4 values\n");
  for(i=0;i<4;i++) {
    scanf("%d",&arr[i]);
  }
  pthread_mutex_lock(&mutex);
  pthread_cond_signal(&cond);
  pthread_mutex_unlock(&mutex);
}


void *read() {
  pthread_mutex_lock(&mutex);
  pthread_cond_wait(&cond, &mutex);
  pthread_mutex_unlock(&mutex);
  printf("Values filled in array are\n");
  int i;
  for(i=0;i<4;i++) {
   printf("%d \n",arr[i]);
  }
  pthread_exit(NULL);
}

int main(void) {
  pthread_t worker_id[2];
  pthread_mutex_init(&mutex, NULL);
  pthread_cond_init(&cond, NULL);
  /*Create thread*/
  pthread_create(&worker_id[0],NULL,&fill,NULL);
  pthread_create(&worker_id[1],NULL,&read,NULL);
  printf("Threads have been created\n");
  /*Wait for all threads*/
  pthread_join(worker_id[0],NULL);
  pthread_join(worker_id[1],NULL);
  return 0;
}
