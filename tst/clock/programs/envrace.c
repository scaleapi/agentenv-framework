/* The mechanism of the Chromium crash, alone: one thread grows and shrinks the environment (setenv/unsetenv, which
 * reallocate environ) while others make timed condvar waits and realtime clock reads, each of which stock libfaketime
 * follows with getenv(). Runs for argv[1] seconds (default 3) and exits 0; a crash shows as a signal. */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static volatile int stop = 0;
static void *mutate(void *arg) {
    (void)arg;
    char name[32], value[64];
    for (unsigned long i = 0; !stop; i++) {
        snprintf(name, sizeof name, "ENVRACE_%lu", i % 512);
        snprintf(value, sizeof value, "%lu-%lu", i, i * 7);
        setenv(name, value, 1);
        if (i % 3 == 0) unsetenv(name);
    }
    return NULL;
}
static void *wait_loop(void *arg) {
    (void)arg;
    pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER; pthread_cond_t c = PTHREAD_COND_INITIALIZER;
    struct timespec ts;
    while (!stop) {
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 1000; if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
        pthread_mutex_lock(&m); pthread_cond_timedwait(&c, &m, &ts); pthread_mutex_unlock(&m);
    }
    return NULL;
}
int main(int argc, char **argv) {
    int seconds = argc > 1 ? atoi(argv[1]) : 3;
    pthread_t t[5];
    pthread_create(&t[0], NULL, mutate, NULL);
    for (int i = 1; i < 5; i++) pthread_create(&t[i], NULL, wait_loop, NULL);
    sleep(seconds);
    stop = 1;
    for (int i = 0; i < 5; i++) pthread_join(t[i], NULL);
    puts("survived");
    return 0;
}
