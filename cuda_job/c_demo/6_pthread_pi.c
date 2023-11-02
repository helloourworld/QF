// Sample program to estimate PI = 3.14159... by randomly generating
// points in the positive quadrant and testing whether they are
// distance <= 1.0 from the origin.  The ratio of "hits" to tries is
// an approximation for the area of the quarter circle with radius 1
// so multiplying the ratio by 4 estimates pi.
//
// usage: picalc <num_samples>
//   num_samples: int, how many sample points to try, higher gets closer to pi
//   num_threads: int, number of threads to use for the computation, default 4

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>


int points_per_thread = -1;
int total_hits = 0;
void compute_pi(void *arg)
{
    int thread_id = (int) arg;
    unsigned int rstate = 123456789 * thread_id;
    int local_hits = 0;
    for (int i = 0; i < points_per_thread; i++)
    {
        double x = ((double) rand_r(&rstate)) / ((double) RAND_MAX);
        double y = ((double) rand_r(&rstate)) / ((double) RAND_MAX);
        if (x * x + y * y <= 1.0)
        {
            local_hits++;
        }
    }
    return (void *) local_hits;
}

int main(int argc, char **argv)
{
    if(argc < 2)
    {
        printf("usage: omp_picalc <num_samples> [num_threads]\n");
        printf("  num_samples: int, how many sample points to try, higher gets closer to pi\n");
        printf("  num_threads: int, number of threads to use for the computation, default 4\n");
        return -1;
    }
    int npoints = atoi(argv[1]);
    int num_threads = argc > 2 ? atoi(argv[2]) : 4;
    points_per_thread = npoints / num_threads;

    pthread_t threads[num_threads];

    for(int p = 0; p < num_threads; p++)
    {
        pthread_create(&threads[p], NULL, compute_pi, (void *) (p + 1));
    }

    for(int p = 0; p < num_threads; p++)
    {
        int local_hits;
        pthread_join(threads[p], (void *) &local_hits);
        total_hits += local_hits;
    }

    double pi_est = ((double)total_hits) / npoints * 4.0;
    printf("npoints: %8d\n", npoints);
    printf("hits:    %8d\n", total_hits);
    printf("pi_est:  %f\n", pi_est);
}