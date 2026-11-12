/* Waits across universe jumps, as the session user: jumpwaits. Reads "arm DIN DAFTER" lines on stdin (seconds of
 * universe time from now); for each it starts one thread per waiter below, prints "ARMED <raw ns>" once they are
 * waiting, and when all have returned prints one JSON line with each waiter's return (CLOCK_MONOTONIC_RAW ns) and
 * result. "in" waiters wait for the clock time now + DIN (a jump the driver makes passes it), "after" ones for now +
 * DAFTER (past the jump: they should end DAFTER - jump real seconds after arming), "rel" ones for 3 s of elapsed time.
 * A reader thread reads CLOCK_MONOTONIC and CLOCK_MONOTONIC_RAW all along and reports backward steps and drift. */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/futex.h>
#include <poll.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/timerfd.h>
#include <time.h>
#include <unistd.h>

static int64_t ns(clockid_t c) { struct timespec t; clock_gettime(c, &t); return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec; }
static struct timespec ts_of(int64_t v) { struct timespec t = {v / 1000000000LL, v % 1000000000LL}; return t; }

enum { IN, AFTER, REL };
struct waiter { const char *name; int kind; int64_t fired; long rc; };
static struct waiter W[] = {
    {"cns_in", IN}, {"tfd_poll_in", IN}, {"tfd_read_in", IN}, {"tfd_epoll_in", IN}, {"ptimer_thread_in", IN},
    {"ptimer_signal_in", IN}, {"cond_rt_in", IN}, {"cond_clock_rt_in", IN}, {"cond_loop_in", IN}, {"sem_rt_in", IN},
    {"futex_rt_in", IN}, {"mutex_rt_in", IN},
    {"cns_after", AFTER}, {"tfd_poll_after", AFTER}, {"cond_loop_after", AFTER}, {"ptimer_thread_after", AFTER},
    {"nanosleep_rel", REL}, {"poll_rel", REL}, {"cond_mono_rel", REL}, {"tfd_mono_rel", REL}, {"cns_mono_rel", REL},
};
#define NW (int)(sizeof W / sizeof W[0])
static int64_t D_in, D_after;
static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER, held = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t c_rt = PTHREAD_COND_INITIALIZER, c_rt2 = PTHREAD_COND_INITIALIZER, c_rt3 = PTHREAD_COND_INITIALIZER,
                      c_rt4 = PTHREAD_COND_INITIALIZER, c_mono;
static sem_t sem;
static volatile int ready = 0, stop_reader = 0;

static void *holder(void *a) { (void)a; pthread_mutex_lock(&held); for (;;) pause(); return NULL; }
static void on_thread_timer(union sigval v) { sem_post((sem_t *)v.sival_ptr); }

static long tfd_wait(int64_t due, int how) {
    int fd = timerfd_create(CLOCK_REALTIME, 0); unsigned long long n = 0;
    struct itimerspec it = {{0, 0}, ts_of(due)};
    if (timerfd_settime(fd, TFD_TIMER_ABSTIME, &it, NULL)) return errno;
    __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST);
    if (how == 1) { struct pollfd p = {fd, POLLIN, 0}; poll(&p, 1, -1); }
    if (how == 2) { int e = epoll_create1(0); struct epoll_event ev = {.events = EPOLLIN}; epoll_ctl(e, EPOLL_CTL_ADD, fd, &ev); epoll_wait(e, &ev, 1, -1); close(e); }
    long r = read(fd, &n, 8) == 8 ? (long)n : -errno;
    close(fd);
    return r;
}
static long ptimer_thread(int64_t due) {
    sem_t s; sem_init(&s, 0, 0); timer_t id;
    struct sigevent sev = {.sigev_notify = SIGEV_THREAD}; sev.sigev_notify_function = on_thread_timer; sev.sigev_value.sival_ptr = &s;
    if (timer_create(CLOCK_REALTIME, &sev, &id)) return -errno;
    struct itimerspec it = {{0, 0}, ts_of(due)};
    if (timer_settime(id, TIMER_ABSTIME, &it, NULL)) return -errno;
    __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST);
    while (sem_wait(&s) != 0) ;
    timer_delete(id);
    return 0;
}
static long cond_loop(pthread_cond_t *c, int64_t due) {
    struct timespec t = ts_of(due); long n = 0;
    pthread_mutex_lock(&m);
    __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST);
    for (;;) { int r = pthread_cond_timedwait(c, &m, &t); n++; if (r == ETIMEDOUT || ns(CLOCK_REALTIME) >= due) break; }
    pthread_mutex_unlock(&m);
    return n;   /* calls it took */
}

static void *run(void *arg) {
    struct waiter *w = arg; long r = 0; struct timespec t; uint32_t word = 0;
    const char *n = w->name;
    if (!strcmp(n, "cns_in") || !strcmp(n, "cns_after")) { t = ts_of(w->kind == IN ? D_in : D_after); __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST); r = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &t, NULL); }
    else if (!strcmp(n, "tfd_poll_in")) r = tfd_wait(D_in, 1);
    else if (!strcmp(n, "tfd_poll_after")) r = tfd_wait(D_after, 1);
    else if (!strcmp(n, "tfd_read_in")) r = tfd_wait(D_in, 0);
    else if (!strcmp(n, "tfd_epoll_in")) r = tfd_wait(D_in, 2);
    else if (!strcmp(n, "ptimer_thread_in")) r = ptimer_thread(D_in);
    else if (!strcmp(n, "ptimer_thread_after")) r = ptimer_thread(D_after);
    else if (!strcmp(n, "ptimer_signal_in")) {
        sigset_t set; sigemptyset(&set); sigaddset(&set, SIGRTMIN + 1); timer_t id; siginfo_t si;
        struct sigevent sev = {.sigev_notify = SIGEV_THREAD_ID, .sigev_signo = SIGRTMIN + 1}; sev._sigev_un._tid = gettid();
        if (timer_create(CLOCK_REALTIME, &sev, &id)) r = -errno;
        else { struct itimerspec it = {{0, 0}, ts_of(D_in)}; timer_settime(id, TIMER_ABSTIME, &it, NULL); __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST);
               r = sigwaitinfo(&set, &si) == SIGRTMIN + 1 ? 0 : -errno; timer_delete(id); }
    }
    else if (!strcmp(n, "cond_rt_in")) { t = ts_of(D_in); pthread_mutex_lock(&m); __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST); r = pthread_cond_timedwait(&c_rt, &m, &t); pthread_mutex_unlock(&m); }
    else if (!strcmp(n, "cond_clock_rt_in")) { t = ts_of(D_in); pthread_mutex_lock(&m); __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST); r = pthread_cond_clockwait(&c_rt2, &m, CLOCK_REALTIME, &t); pthread_mutex_unlock(&m); }
    else if (!strcmp(n, "cond_loop_in")) r = cond_loop(&c_rt3, D_in);
    else if (!strcmp(n, "cond_loop_after")) r = cond_loop(&c_rt4, D_after);
    else if (!strcmp(n, "sem_rt_in")) { t = ts_of(D_in); __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST); r = sem_timedwait(&sem, &t) ? errno : 0; }
    else if (!strcmp(n, "futex_rt_in")) { t = ts_of(D_in); __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST); r = syscall(SYS_futex, &word, FUTEX_WAIT_BITSET_PRIVATE | FUTEX_CLOCK_REALTIME, 0, &t, NULL, FUTEX_BITSET_MATCH_ANY) ? errno : 0; }
    else if (!strcmp(n, "mutex_rt_in")) { t = ts_of(D_in); __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST); r = pthread_mutex_timedlock(&held, &t); }
    else if (!strcmp(n, "nanosleep_rel")) { t = ts_of(3000000000LL); __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST); r = nanosleep(&t, NULL); }
    else if (!strcmp(n, "poll_rel")) { __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST); r = poll(NULL, 0, 3000); }
    else if (!strcmp(n, "cond_mono_rel")) { t = ts_of(ns(CLOCK_MONOTONIC) + 3000000000LL); pthread_mutex_lock(&m); __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST); r = pthread_cond_timedwait(&c_mono, &m, &t); pthread_mutex_unlock(&m); }
    else if (!strcmp(n, "tfd_mono_rel")) { int fd = timerfd_create(CLOCK_MONOTONIC, 0); struct itimerspec it = {{0, 0}, ts_of(3000000000LL)}; unsigned long long k; timerfd_settime(fd, 0, &it, NULL); __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST); r = read(fd, &k, 8) == 8 ? 0 : -errno; close(fd); }
    else if (!strcmp(n, "cns_mono_rel")) { t = ts_of(3000000000LL); __atomic_add_fetch(&ready, 1, __ATOMIC_SEQ_CST); r = clock_nanosleep(CLOCK_MONOTONIC, 0, &t, NULL); }
    w->fired = ns(CLOCK_MONOTONIC_RAW);
    w->rc = r;
    return NULL;
}

static long mono_back; static int64_t mono_back_max, drift_min = INT64_MAX, drift_max = INT64_MIN;
static void *reader(void *a) {
    (void)a; int64_t last = ns(CLOCK_MONOTONIC), off0 = ns(CLOCK_MONOTONIC) - ns(CLOCK_MONOTONIC_RAW);
    for (long i = 0; !stop_reader; i++) {
        int64_t mo = ns(CLOCK_MONOTONIC);
        if (mo < last) { mono_back++; if (last - mo > mono_back_max) mono_back_max = last - mo; }
        last = mo;
        if ((i & 1023) == 0) { int64_t d = mo - ns(CLOCK_MONOTONIC_RAW) - off0; if (d < drift_min) drift_min = d; if (d > drift_max) drift_max = d; }
    }
    return NULL;
}

int main(void) {
    pthread_condattr_t at; pthread_condattr_init(&at); pthread_condattr_setclock(&at, CLOCK_MONOTONIC); pthread_cond_init(&c_mono, &at);
    sem_init(&sem, 0, 0);
    sigset_t set; sigemptyset(&set); sigaddset(&set, SIGRTMIN + 1); pthread_sigmask(SIG_BLOCK, &set, NULL);
    pthread_t h, rd; pthread_create(&h, NULL, holder, NULL); pthread_create(&rd, NULL, reader, NULL);
    while (pthread_mutex_trylock(&held) == 0) { pthread_mutex_unlock(&held); sched_yield(); }
    setvbuf(stdout, NULL, _IOLBF, 0);
    char line[256]; int round = 0; int64_t raw_start = ns(CLOCK_MONOTONIC_RAW), mono_start = ns(CLOCK_MONOTONIC);
    while (fgets(line, sizeof line, stdin)) {
        double din, dafter;
        if (sscanf(line, "arm %lf %lf", &din, &dafter) != 2) break;
        int64_t u0 = ns(CLOCK_REALTIME), t0 = ns(CLOCK_MONOTONIC_RAW);
        D_in = u0 + (int64_t)(din * 1e9); D_after = u0 + (int64_t)(dafter * 1e9);
        ready = 0;
        pthread_t t[64];
        for (int i = 0; i < NW; i++) { W[i].fired = 0; W[i].rc = -999; pthread_create(&t[i], NULL, run, &W[i]); }
        while (__atomic_load_n(&ready, __ATOMIC_SEQ_CST) < NW) usleep(1000);
        usleep(50000);
        printf("ARMED %lld %lld\n", (long long)t0, (long long)ns(CLOCK_MONOTONIC_RAW));
        for (int i = 0; i < NW; i++) pthread_join(t[i], NULL);
        printf("{\"round\": %d, \"t0\": %lld, \"u0\": %lld, \"w\": {", round++, (long long)t0, (long long)u0);
        for (int i = 0; i < NW; i++) printf("%s\"%s\": [%d, %lld, %ld]", i ? ", " : "", W[i].name, W[i].kind, (long long)W[i].fired, W[i].rc);
        printf("}, \"mono_back\": %ld, \"mono_back_max_ns\": %lld, \"mono_minus_raw_drift_ns\": [%lld, %lld], \"mono_rate\": %.9f}\n",
               mono_back, (long long)mono_back_max, (long long)drift_min, (long long)drift_max,
               (double)(ns(CLOCK_MONOTONIC) - mono_start) / (double)(ns(CLOCK_MONOTONIC_RAW) - raw_start));
    }
    stop_reader = 1; pthread_join(rd, NULL);
    return 0;
}
