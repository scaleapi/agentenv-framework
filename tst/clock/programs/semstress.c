/* sem_timedwait (CLOCK_REALTIME deadline 1 h ahead) against sem_post: semstress SECONDS GAP_US. A consumer takes posts
 * with sem_timedwait; a producer posts every GAP_US (+-20 %) once the last post was taken, so the consumer is asleep in
 * the futex wait for most posts. Prints posts, takes, signal-to-take latency (CLOCK_MONOTONIC_RAW) and errors. */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static sem_t s; static volatile int64_t sent; static volatile long taken; static volatile int stop; static int64_t lat[2000000]; static int nlat; static long errs, timeouts;
static int64_t raw(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec; }
static void *consumer(void *a) {
    (void)a;
    while (!stop) {
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_sec += 3600;
        if (sem_timedwait(&s, &ts) == 0) { if (nlat < 2000000) lat[nlat++] = raw() - sent; __atomic_add_fetch(&taken, 1, __ATOMIC_SEQ_CST); }
        else if (errno == ETIMEDOUT) timeouts++; else if (errno != EINTR) errs++;
    }
    return NULL;
}
static int cmp(const void *a, const void *b) { int64_t x = *(int64_t *)a, y = *(int64_t *)b; return x < y ? -1 : x > y; }
int main(int argc, char **argv) {
    double secs = atof(argv[1]); long gap = atol(argv[2]); pthread_t t; sem_init(&s, 0, 0); pthread_create(&t, NULL, consumer, NULL);
    int64_t end = raw() + (int64_t)(secs * 1e9); unsigned seed = 7; long posts = 0, stuck = 0;
    while (raw() < end) {
        long us = gap + (long)((rand_r(&seed) % 401 - 200) / 1000.0 * gap);
        struct timespec d = {us / 1000000, (us % 1000000) * 1000}; clock_nanosleep(CLOCK_MONOTONIC, 0, &d, NULL);
        sent = raw(); sem_post(&s); posts++;
        int64_t w0 = raw(); while (__atomic_load_n(&taken, __ATOMIC_SEQ_CST) < posts) { if (raw() - w0 > 2000000000LL) { stuck++; break; } sched_yield(); }
    }
    stop = 1; sem_post(&s); pthread_join(t, NULL);
    qsort(lat, nlat, sizeof lat[0], cmp);
    int over = 0; for (int i = 0; i < nlat; i++) if (lat[i] > 50000000) over++;
    printf("{\"posts\": %ld, \"taken\": %ld, \"stuck_2s\": %ld, \"max_ms\": %.3f, \"p999_ms\": %.3f, \"p50_ms\": %.4f, \"over_50ms\": %d, \"timeouts\": %ld, \"errors\": %ld}\n",
           posts, taken - 1 < posts ? taken : taken - 1, stuck, nlat ? lat[nlat - 1] / 1e6 : 0, nlat ? lat[(int)(nlat * 0.999)] / 1e6 : 0, nlat ? lat[nlat / 2] / 1e6 : 0, over, timeouts, errs);
    return 0;
}
