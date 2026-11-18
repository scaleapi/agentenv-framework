/* Realtime condvar waits under signals and clock changes.
 *   condstress2 stress SECONDS GAP_US   a waiter waits for a flag with a 1 h CLOCK_REALTIME deadline (predicate loop,
 *        the deadline recomputed from the clock each time round); a signaler sets the flag and signals every GAP_US
 *        (+-20 %) once the last one was taken. Prints handoffs, signal-to-take latency (CLOCK_MONOTONIC_RAW) and the
 *        waiter's returns that found no flag (spurious) or timed out.
 *   condstress2 single N SECONDS        N threads each make one pthread_cond_timedwait with a deadline SECONDS of clock
 *        time ahead and no signal; prints each one's result and real seconds. */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "rawsys.h"
static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t c = PTHREAD_COND_INITIALIZER;
static int flag = 0, stop = 0; static int64_t sent; static int64_t lat[4000000]; static int nlat; static long spurious, timeouts, calls;
static int64_t raw(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec; }
static void *waiter(void *a) {
    (void)a;
    pthread_mutex_lock(&m);
    while (!stop) {
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_sec += 3600;
        while (!flag && !stop) { int r = pthread_cond_timedwait(&c, &m, &ts); calls++; if (r == ETIMEDOUT) { timeouts++; break; } if (!flag && !stop) spurious++; }
        if (flag) { if (nlat < 4000000) lat[nlat++] = raw() - sent; flag = 0; }
    }
    pthread_mutex_unlock(&m);
    return NULL;
}
static int cmp(const void *a, const void *b) { int64_t x = *(int64_t *)a, y = *(int64_t *)b; return x < y ? -1 : x > y; }
static double single_s; static pthread_cond_t sc = PTHREAD_COND_INITIALIZER;
struct sres { int rc; double real; };
static void *single(void *a) {
    struct sres *r = a; struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    long long d = (long long)ts.tv_nsec + (long long)(single_s * 1e9); ts.tv_sec += d / 1000000000LL; ts.tv_nsec = d % 1000000000LL;
    int64_t t0 = raw(); pthread_mutex_lock(&m); r->rc = pthread_cond_timedwait(&sc, &m, &ts); pthread_mutex_unlock(&m); r->real = (raw() - t0) / 1e9;
    return NULL;
}
int main(int argc, char **argv) {
    if (!strcmp(argv[1], "single")) {
        int n = atoi(argv[2]); single_s = atof(argv[3]); pthread_t t[256]; struct sres r[256];
        for (int i = 0; i < n; i++) pthread_create(&t[i], NULL, single, &r[i]);
        for (int i = 0; i < n; i++) pthread_join(t[i], NULL);
        printf("{\"n\": %d, \"seconds\": %g, \"results\": [", n, single_s);
        for (int i = 0; i < n; i++) printf("%s[\"%s\", %.4f]", i ? ", " : "", r[i].rc == ETIMEDOUT ? "ETIMEDOUT" : r[i].rc == 0 ? "0" : "other", r[i].real);
        printf("]}\n");
        return 0;
    }
    double s = atof(argv[2]); long gap = atol(argv[3]); pthread_t t; pthread_create(&t, NULL, waiter, NULL);
    int64_t end = raw() + (int64_t)(s * 1e9); unsigned seed = 1;
    while (raw() < end) {
        long us = gap + (long)((rand_r(&seed) % 401 - 200) / 1000.0 * gap);
        struct timespec d = {us / 1000000, (us % 1000000) * 1000};
        raw_nanosleep(&d);
        pthread_mutex_lock(&m);
        if (!flag) { flag = 1; sent = raw(); pthread_cond_signal(&c); }
        pthread_mutex_unlock(&m);
        int64_t w0 = raw();
        for (;;) { pthread_mutex_lock(&m); int f = flag; pthread_mutex_unlock(&m); if (!f || raw() - w0 > 2000000000LL) break; sched_yield(); }
    }
    pthread_mutex_lock(&m); stop = 1; pthread_cond_signal(&c); pthread_mutex_unlock(&m);
    pthread_join(t, NULL);
    qsort(lat, nlat, sizeof lat[0], cmp);
    int over = 0; for (int i = 0; i < nlat; i++) if (lat[i] > 50000000) over++;
    printf("{\"handoffs\": %d, \"max_ms\": %.3f, \"p999_ms\": %.3f, \"p50_ms\": %.4f, \"over_50ms\": %d, \"spurious\": %ld, \"timeouts\": %ld, \"calls\": %ld}\n", nlat,
           nlat ? lat[nlat - 1] / 1e6 : 0, nlat ? lat[(int)(nlat * 0.999)] / 1e6 : 0, nlat ? lat[nlat / 2] / 1e6 : 0, over, spurious, timeouts, calls);
    return 0;
}
