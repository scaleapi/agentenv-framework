/* Torn page reads, black box: torn SECONDS THREADS OFF_A_S OFF_B_S. The driver rewrites the clock page as fast as it can,
 * alternating a universe OFF_A_S and one OFF_B_S from real time, both at rate 1, while each thread here reads
 * CLOCK_REALTIME through the library as fast as it can. Real time is the raw system call's CLOCK_REALTIME at the start
 * plus CLOCK_MONOTONIC_RAW since, neither of which the library touches; a read more than 1 s from both offsets mixes two
 * pages, so it's torn. Prints one JSON line. */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "rawsys.h"
#define NS 1000000000LL
static double seconds; static int64_t off_a, off_b, rt0, raw0;
static int64_t ns(clockid_t c) { struct timespec t; clock_gettime(c, &t); return (int64_t)t.tv_sec * NS + t.tv_nsec; }
static int64_t far(int64_t d) { return d < 0 ? -d : d; }
struct res { long reads, a, b, torn; int64_t worst; };
static struct res R[16];
static void *reader(void *arg) {
    struct res *r = arg;
    int64_t end = ns(CLOCK_MONOTONIC_RAW) + (int64_t)(seconds * NS);
    for (;;) {
        int64_t raw = ns(CLOCK_MONOTONIC_RAW), off = ns(CLOCK_REALTIME) - (rt0 + raw - raw0);
        r->reads++;
        if (far(off - off_a) <= NS) r->a++;
        else if (far(off - off_b) <= NS) r->b++;
        else { r->torn++; if (far(off) > far(r->worst)) r->worst = off; }
        if (raw >= end) break;
    }
    return NULL;
}
int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: torn SECONDS THREADS OFF_A_S OFF_B_S\n"); return 2; }
    seconds = atof(argv[1]); int n = atoi(argv[2]); off_a = (int64_t)(atof(argv[3]) * NS); off_b = (int64_t)(atof(argv[4]) * NS);
    if (n < 1 || n > 16) return 2;
    struct timespec t; raw_clock_gettime(CLOCK_REALTIME, &t); raw0 = ns(CLOCK_MONOTONIC_RAW); rt0 = (int64_t)t.tv_sec * NS + t.tv_nsec;
    pthread_t th[16];
    for (int i = 0; i < n; i++) pthread_create(&th[i], NULL, reader, &R[i]);
    long reads = 0, a = 0, b = 0, torn = 0; int64_t worst = 0;
    for (int i = 0; i < n; i++) {
        pthread_join(th[i], NULL);
        reads += R[i].reads; a += R[i].a; b += R[i].b; torn += R[i].torn;
        if (far(R[i].worst) > far(worst)) worst = R[i].worst;
    }
    printf("{\"reads\": %ld, \"universe_a\": %ld, \"universe_b\": %ld, \"torn\": %ld, \"worst_torn_offset_s\": %.3f}\n",
           reads, a, b, torn, (double)worst / NS);
    return 0;
}
