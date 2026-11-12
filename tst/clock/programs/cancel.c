/* Waits for a clock time cancelled: a thread cancelled in sem_timedwait or clock_nanosleep goes at once, and the
   semaphore keeps no waiter for it; 600 threads cancelled in pthread_cond_timedwait, more than the library's table of
   condvar waits holds, leave it as they found it, which the wait after them shows by ending at the jump past its
   deadline. A realtime futex wait on an unaligned word fails at once, as the kernel's does. Prints {"sem_cancelled",
   "sem_waiters", "cns_cancelled", "cond_cancelled", "misaligned_errno", "misaligned_ms"}, then READY with the last wait
   in progress, then {"fired_raw", "rc"} once it ends. */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/futex.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define CONDS 600

static int64_t ns(clockid_t c) { struct timespec t; clock_gettime(c, &t); return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec; }
static struct timespec ts_of(int64_t v) { struct timespec t = {v / 1000000000LL, v % 1000000000LL}; return t; }
static struct timespec hour_ahead(void) { return ts_of(ns(CLOCK_REALTIME) + 3600000000000LL); }

static sem_t sem;
static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t c = PTHREAD_COND_INITIALIZER;
static volatile int entered;
static int64_t fired_raw;
static long last_rc;

static void unlock(void *mutex) { pthread_mutex_unlock(mutex); }
static void *sem_waiter(void *a) { struct timespec t = hour_ahead(); (void)a; entered = 1; sem_timedwait(&sem, &t); return NULL; }
static void *cns_waiter(void *a) { struct timespec t = hour_ahead(); (void)a; entered = 1; clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &t, NULL); return NULL; }
static void *cond_waiter(void *a) {
    struct timespec t = hour_ahead(); long r;
    pthread_mutex_lock(&m);
    pthread_cleanup_push(unlock, &m);
    entered = 1;
    r = pthread_cond_timedwait(&c, &m, &t);
    pthread_cleanup_pop(1);
    if (a) { fired_raw = ns(CLOCK_MONOTONIC_RAW); last_rc = r; }
    return NULL;
}

/* a thread started on fn, in its wait: a condvar waiter once it has released the mutex, others after 50 ms */
static pthread_t started(void *(*fn)(void *), void *arg) {
    pthread_t t;
    entered = 0;
    pthread_create(&t, NULL, fn, arg);
    while (!entered) usleep(100);
    if (fn == cond_waiter) { pthread_mutex_lock(&m); pthread_mutex_unlock(&m); }
    else usleep(50000);
    return t;
}
/* joined within ``seconds`` of real time, and how it ended: 1 cancelled, 0 not */
static int joined(pthread_t t, int seconds, void **res) {
    struct timespec dl = ts_of(ns(CLOCK_MONOTONIC) + seconds * 1000000000LL);
    return pthread_clockjoin_np(t, res, CLOCK_MONOTONIC, &dl) == 0;
}
static int cancelled(void *(*fn)(void *)) {
    void *res = NULL;
    pthread_t t = started(fn, NULL);
    pthread_cancel(t);
    return joined(t, 2, &res) && res == PTHREAD_CANCELED;
}

/* a realtime futex wait an hour ahead on a word one byte off its alignment: its errno, and how long it took */
static long misaligned(double *ms) {
    static uint32_t words[2];
    struct timespec t = hour_ahead();
    int64_t t0 = ns(CLOCK_MONOTONIC);
    long r = syscall(SYS_futex, (char *)words + 1, FUTEX_WAIT_BITSET_PRIVATE | FUTEX_CLOCK_REALTIME, 0, &t, NULL,
                     FUTEX_BITSET_MATCH_ANY);
    *ms = (ns(CLOCK_MONOTONIC) - t0) / 1e6;
    return r == -1 ? errno : 0;
}

int main(void) {
    int sem_ok, cns_ok, cond_ok = 0;
    uint64_t waiters;
    double ms;
    long bad = misaligned(&ms);
    void *res;
    sem_init(&sem, 0, 0);
    sem_ok = cancelled(sem_waiter);
    waiters = *(volatile uint64_t *)&sem >> 32;   /* glibc's semaphore: its waiters in the high word */
    cns_ok = cancelled(cns_waiter);
    for (int i = 0; i < CONDS; i++) cond_ok += cancelled(cond_waiter);
    printf("{\"sem_cancelled\": %d, \"sem_waiters\": %llu, \"cns_cancelled\": %d, \"cond_cancelled\": %d, "
           "\"misaligned_errno\": %ld, \"misaligned_ms\": %.3f}\n", sem_ok, (unsigned long long)waiters, cns_ok, cond_ok,
           bad, ms);
    pthread_t last = started(cond_waiter, (void *)1);
    printf("READY\n");
    fflush(stdout);
    if (!joined(last, 10, &res)) { printf("{\"fired_raw\": 0, \"rc\": -1}\n"); return 0; }
    printf("{\"fired_raw\": %lld, \"rc\": %ld}\n", (long long)fired_raw, last_rc);
    return 0;
}
