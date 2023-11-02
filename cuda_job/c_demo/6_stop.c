#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
void *RTPfun(void * client_addr);

int main(int argc, char *argv[])
{
   pthread_t RTPthread;
   int client_addr;
   for (int i; i<5; i++){
   pthread_create(&RTPthread, NULL, &RTPfun, (void*)client_addr);
   sleep(2);
}
   pthread_cancel(RTPthread);
   pthread_join(RTPthread, NULL);

   return 0;
 }

 void *RTPfun(void * client_addr)
 {
    int count = 0;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
    while(1) {
        if(count > 10) {
                printf("thread set for cancel\n");
                pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);     
        }
        sleep(1);
        printf("count:%d\n", count);
        count ++;
    }
    return 0;
 }