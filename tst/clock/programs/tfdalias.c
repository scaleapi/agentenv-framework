/* A timerfd's tracking follows its file, not its number: A, armed for a clock time an hour ahead, is duplicated as A2
   (by METHOD: dup, dup2, dup3, fcntl's F_DUPFD_CLOEXEC, or dup then close_range) and closed, and A's number goes to a
   duplicate of O, a monotonic timerfd armed 10 s out. After READY the driver jumps 2 h: A2 must fire at the jump and O,
   now on A's old number, must not fire early. Prints READY, then {"same_number", "a2_fired_raw", "a2_count",
   "o_fired_early"}. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/timerfd.h>
#include <time.h>
#include <unistd.h>

static int64_t ns(clockid_t c) { struct timespec t; clock_gettime(c, &t); return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec; }
static struct timespec ts_of(int64_t v) { struct timespec t = {v / 1000000000LL, v % 1000000000LL}; return t; }

int main(int argc, char **argv) {
    const char *how = argc > 1 ? argv[1] : "dup";
    int o = timerfd_create(CLOCK_MONOTONIC, 0), a = timerfd_create(CLOCK_REALTIME, 0), a2, o2;
    struct itimerspec ten = {{0, 0}, ts_of(10000000000LL)}, hour = {{0, 0}, ts_of(ns(CLOCK_REALTIME) + 3600000000000LL)};
    unsigned long long count = 0;
    int64_t fired = 0;
    timerfd_settime(o, 0, &ten, NULL);
    timerfd_settime(a, TFD_TIMER_ABSTIME, &hour, NULL);
    if (!strcmp(how, "dup2")) a2 = dup2(a, 100);
    else if (!strcmp(how, "dup3")) a2 = dup3(a, 101, O_CLOEXEC);
    else if (!strcmp(how, "fcntl")) a2 = fcntl(a, F_DUPFD_CLOEXEC, 50);
    else a2 = dup(a);
    if (!strcmp(how, "close_range")) close_range(a, a, 0);
    else close(a);
    o2 = dup(o);
    printf("READY\n");
    fflush(stdout);
    struct pollfd p = {a2, POLLIN, 0};
    if (poll(&p, 1, 5000) == 1 && read(a2, &count, 8) == 8) fired = ns(CLOCK_MONOTONIC_RAW);
    struct pollfd q = {o2, POLLIN, 0};
    int early = poll(&q, 1, 1000) == 1;
    printf("{\"same_number\": %d, \"a2_fired_raw\": %lld, \"a2_count\": %llu, \"o_fired_early\": %d}\n", o2 == a,
           (long long)fired, count, early);
    return 0;
}
