/* Tight clock readers: rtread SECONDS THREADS OUT. Each thread reads CLOCK_REALTIME and CLOCK_MONOTONIC as fast as it
 * can for SECONDS (CLOCK_MONOTONIC_RAW) and counts reads that went back, with the largest step back; the first thread
 * also gives CLOCK_MONOTONIC's rate against CLOCK_MONOTONIC_RAW over the run. Writes JSON to OUT. */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double dur;
static int64_t ns(clockid_t c) { struct timespec t; clock_gettime(c, &t); return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec; }
struct res { long reads, rt_back, mono_back; int64_t rt_max_back, mono_max_back; double mono_rate; };
static struct res R[16];
static void *reader(void *arg) {
    struct res *r = arg;
    int64_t raw0 = ns(CLOCK_MONOTONIC_RAW), mono0 = ns(CLOCK_MONOTONIC), end = raw0 + (int64_t)(dur * 1e9);
    int64_t lrt = ns(CLOCK_REALTIME), lmo = mono0;
    for (;;) {
        int64_t rt = ns(CLOCK_REALTIME), mo = ns(CLOCK_MONOTONIC);
        r->reads++;
        if (rt < lrt) { r->rt_back++; if (lrt - rt > r->rt_max_back) r->rt_max_back = lrt - rt; }
        if (mo < lmo) { r->mono_back++; if (lmo - mo > r->mono_max_back) r->mono_max_back = lmo - mo; }
        lrt = rt; lmo = mo;
        if ((r->reads & 255) == 0 && ns(CLOCK_MONOTONIC_RAW) >= end) break;
    }
    r->mono_rate = (double)(ns(CLOCK_MONOTONIC) - mono0) / (double)(ns(CLOCK_MONOTONIC_RAW) - raw0);
    return NULL;
}
int main(int argc, char **argv) {
    dur = atof(argv[1]); int n = atoi(argv[2]); pthread_t t[16];
    for (int i = 0; i < n; i++) pthread_create(&t[i], NULL, reader, &R[i]);
    for (int i = 0; i < n; i++) pthread_join(t[i], NULL);
    FILE *f = fopen(argv[3], "w");
    fprintf(f, "{\"threads\": [");
    for (int i = 0; i < n; i++) fprintf(f, "%s{\"reads\": %ld, \"rt_back\": %ld, \"rt_max_back_ns\": %lld, \"mono_back\": %ld, \"mono_max_back_ns\": %lld, \"mono_rate\": %.9f}", i ? ", " : "",
            R[i].reads, R[i].rt_back, (long long)R[i].rt_max_back, R[i].mono_back, (long long)R[i].mono_max_back, R[i].mono_rate);
    fprintf(f, "]}\n"); fclose(f);
    return 0;
}
