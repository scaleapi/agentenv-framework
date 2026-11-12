/* One wait of SECONDS of the program's own (universe) time through API, in this process:
 *   uwait API SECONDS
 * prints one JSON line: real seconds it took (CLOCK_MONOTONIC_RAW, which the library leaves real), the universe's
 * realtime and monotonic seconds it covered, and the call's result. */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/futex.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/select.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/timerfd.h>
#include <time.h>
#include <unistd.h>

static double secs(clockid_t c) { struct timespec t; clock_gettime(c, &t); return t.tv_sec + t.tv_nsec / 1e9; }
static struct timespec plus(struct timespec t, double s) {
    long long ns = (long long)(s * 1e9) + t.tv_nsec;
    t.tv_sec += ns / 1000000000LL; t.tv_nsec = ns % 1000000000LL;
    return t;
}
static struct timespec clk(clockid_t c, double s) { struct timespec t; clock_gettime(c, &t); return plus(t, s); }
static struct timespec rel(double s) { struct timespec t = {0, 0}; return plus(t, s); }
static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER, held = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t c_rt = PTHREAD_COND_INITIALIZER, c_mono;
static sem_t sem;
static volatile int got_alarm = 0;
static void on_alarm(int s) { (void)s; got_alarm = 1; }
static sem_t tsem;
static void on_ptimer(union sigval v) { (void)v; sem_post(&tsem); }
static void *holder(void *a) { (void)a; pthread_mutex_lock(&held); pause(); return NULL; }
static void hold(void) { pthread_t t; pthread_create(&t, NULL, holder, NULL); while (pthread_mutex_trylock(&held) == 0) { pthread_mutex_unlock(&held); sched_yield(); } }
static long tfd_wait(int fd, int use_poll) {
    unsigned long long n = 0;
    if (use_poll) { struct pollfd p = {fd, POLLIN, 0}; if (poll(&p, 1, -1) < 0) return errno; }
    return read(fd, &n, 8) == 8 ? 0 : errno;
}

static long run(const char *a, double s) {
    struct timespec ts;
    uint32_t word = 0;
    if (!strcmp(a, "nanosleep")) { ts = rel(s); return nanosleep(&ts, NULL) ? errno : 0; }
    if (!strcmp(a, "clock_nanosleep_rt_rel")) { ts = rel(s); return clock_nanosleep(CLOCK_REALTIME, 0, &ts, NULL); }
    if (!strcmp(a, "clock_nanosleep_mono_rel")) { ts = rel(s); return clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, NULL); }
    if (!strcmp(a, "clock_nanosleep_mono_abs")) { ts = clk(CLOCK_MONOTONIC, s); return clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL); }
    if (!strcmp(a, "clock_nanosleep_rt_abs")) { ts = clk(CLOCK_REALTIME, s); return clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &ts, NULL); }
    if (!strcmp(a, "clock_nanosleep_boot_abs")) { ts = clk(CLOCK_BOOTTIME, s); return clock_nanosleep(CLOCK_BOOTTIME, TIMER_ABSTIME, &ts, NULL); }
    if (!strcmp(a, "usleep")) { return usleep((useconds_t)(s * 1e6)) ? errno : 0; }
    if (!strcmp(a, "sleep")) { return sleep((unsigned)s); }
    if (!strcmp(a, "select")) { ts = rel(s); struct timeval tv = {ts.tv_sec, ts.tv_nsec / 1000}; return select(0, NULL, NULL, NULL, &tv) < 0 ? errno : 0; }
    if (!strcmp(a, "pselect")) { ts = rel(s); return pselect(0, NULL, NULL, NULL, &ts, NULL) < 0 ? errno : 0; }
    if (!strcmp(a, "poll")) { return poll(NULL, 0, (int)(s * 1000)) < 0 ? errno : 0; }
    if (!strcmp(a, "ppoll")) { ts = rel(s); return ppoll(NULL, 0, &ts, NULL) < 0 ? errno : 0; }
    if (!strcmp(a, "epoll_wait")) { int e = epoll_create1(0); struct epoll_event ev; return epoll_wait(e, &ev, 1, (int)(s * 1000)) < 0 ? errno : 0; }
    if (!strcmp(a, "epoll_pwait")) { int e = epoll_create1(0); struct epoll_event ev; return epoll_pwait(e, &ev, 1, (int)(s * 1000), NULL) < 0 ? errno : 0; }
    if (!strcmp(a, "epoll_pwait2")) { int e = epoll_create1(0); struct epoll_event ev; ts = rel(s); return epoll_pwait2(e, &ev, 1, &ts, NULL) < 0 ? errno : 0; }
    if (!strncmp(a, "timerfd_", 8)) {
        int mono = strstr(a, "_mono") != NULL, abs_ = strstr(a, "_abs") != NULL, by_poll = strstr(a, "_read") == NULL;
        int fd = timerfd_create(mono ? CLOCK_MONOTONIC : CLOCK_REALTIME, 0);
        struct itimerspec it = {{0, 0}, abs_ ? clk(mono ? CLOCK_MONOTONIC : CLOCK_REALTIME, s) : rel(s)};
        if (timerfd_settime(fd, abs_ ? TFD_TIMER_ABSTIME : 0, &it, NULL)) return errno;
        return tfd_wait(fd, by_poll);
    }
    if (!strcmp(a, "cond_timedwait_rt")) { ts = clk(CLOCK_REALTIME, s); pthread_mutex_lock(&m); return pthread_cond_timedwait(&c_rt, &m, &ts); }
    if (!strcmp(a, "cond_timedwait_mono")) { ts = clk(CLOCK_MONOTONIC, s); pthread_mutex_lock(&m); return pthread_cond_timedwait(&c_mono, &m, &ts); }
    if (!strcmp(a, "cond_clockwait_mono")) { ts = clk(CLOCK_MONOTONIC, s); pthread_mutex_lock(&m); return pthread_cond_clockwait(&c_rt, &m, CLOCK_MONOTONIC, &ts); }
    if (!strcmp(a, "cond_clockwait_rt")) { ts = clk(CLOCK_REALTIME, s); pthread_mutex_lock(&m); return pthread_cond_clockwait(&c_rt, &m, CLOCK_REALTIME, &ts); }
    if (!strcmp(a, "cond_timedwait_rt_loop") || !strcmp(a, "cond_timedwait_mono_loop")) {
        /* as a caller with a predicate does: a wakeup before the deadline (by its own clock) waits again */
        int mono = strstr(a, "mono") != NULL; clockid_t c = mono ? CLOCK_MONOTONIC : CLOCK_REALTIME; struct timespec now;
        ts = clk(c, s); pthread_mutex_lock(&m);
        for (;;) { int r = pthread_cond_timedwait(mono ? &c_mono : &c_rt, &m, &ts); clock_gettime(c, &now);
                   if (r == ETIMEDOUT || now.tv_sec > ts.tv_sec || (now.tv_sec == ts.tv_sec && now.tv_nsec >= ts.tv_nsec)) return 0; }
    }
    if (!strcmp(a, "sem_timedwait")) { ts = clk(CLOCK_REALTIME, s); return sem_timedwait(&sem, &ts) ? errno : 0; }
    if (!strcmp(a, "sem_clockwait_mono")) { ts = clk(CLOCK_MONOTONIC, s); return sem_clockwait(&sem, CLOCK_MONOTONIC, &ts) ? errno : 0; }
    if (!strcmp(a, "mutex_timedlock")) { hold(); ts = clk(CLOCK_REALTIME, s); return pthread_mutex_timedlock(&held, &ts); }
    if (!strcmp(a, "mutex_clocklock_mono")) { hold(); ts = clk(CLOCK_MONOTONIC, s); return pthread_mutex_clocklock(&held, CLOCK_MONOTONIC, &ts); }
    if (!strcmp(a, "futex_wait_rel")) { ts = rel(s); return syscall(SYS_futex, &word, FUTEX_WAIT_PRIVATE, 0, &ts, NULL, 0) ? errno : 0; }
    if (!strcmp(a, "futex_bitset_mono_abs")) { ts = clk(CLOCK_MONOTONIC, s); return syscall(SYS_futex, &word, FUTEX_WAIT_BITSET_PRIVATE, 0, &ts, NULL, FUTEX_BITSET_MATCH_ANY) ? errno : 0; }
    if (!strcmp(a, "alarm")) { signal(SIGALRM, on_alarm); alarm((unsigned)s); while (!got_alarm) pause(); return 0; }
    /* waits added for the clock-time APIs and timers */
    if (!strcmp(a, "setitimer_real")) { struct itimerval v = {{0, 0}, {(long)s, (long)((s - (long)s) * 1e6)}}; signal(SIGALRM, on_alarm); setitimer(ITIMER_REAL, &v, NULL); while (!got_alarm) pause(); return 0; }
    if (!strcmp(a, "timerfd_rt_abs_read") || !strcmp(a, "timerfd_rt_abs_epoll")) {
        int fd = timerfd_create(CLOCK_REALTIME, 0);
        struct itimerspec it = {{0, 0}, clk(CLOCK_REALTIME, s)};
        if (timerfd_settime(fd, TFD_TIMER_ABSTIME, &it, NULL)) return errno;
        if (strstr(a, "epoll")) { int e = epoll_create1(0); struct epoll_event ev = {.events = EPOLLIN}; epoll_ctl(e, EPOLL_CTL_ADD, fd, &ev); if (epoll_wait(e, &ev, 1, -1) != 1) return errno; }
        unsigned long long n = 0; return read(fd, &n, 8) == 8 ? 0 : errno;
    }
    if (!strcmp(a, "ptimer_rt_abs_signal") || !strcmp(a, "ptimer_mono_rel_signal")) {
        int rt = strstr(a, "_rt_") != NULL; sigset_t set; sigemptyset(&set); sigaddset(&set, SIGRTMIN); sigprocmask(SIG_BLOCK, &set, NULL);
        struct sigevent sev = {.sigev_notify = SIGEV_SIGNAL, .sigev_signo = SIGRTMIN}; timer_t id;
        if (timer_create(rt ? CLOCK_REALTIME : CLOCK_MONOTONIC, &sev, &id)) return errno;
        struct itimerspec it = {{0, 0}, rt ? clk(CLOCK_REALTIME, s) : rel(s)};
        if (timer_settime(id, rt ? TIMER_ABSTIME : 0, &it, NULL)) return errno;
        siginfo_t si; return sigwaitinfo(&set, &si) == SIGRTMIN ? 0 : errno;
    }
    if (!strcmp(a, "ptimer_rt_abs_thread")) {
        struct sigevent sev = {.sigev_notify = SIGEV_THREAD}; sev.sigev_notify_function = on_ptimer; timer_t id;
        if (timer_create(CLOCK_REALTIME, &sev, &id)) return errno;
        struct itimerspec it = {{0, 0}, clk(CLOCK_REALTIME, s)};
        if (timer_settime(id, TIMER_ABSTIME, &it, NULL)) return errno;
        while (sem_wait(&tsem) != 0) ;
        return 0;
    }
    if (!strcmp(a, "futex_bitset_rt_abs")) { ts = clk(CLOCK_REALTIME, s); return syscall(SYS_futex, &word, FUTEX_WAIT_BITSET_PRIVATE | FUTEX_CLOCK_REALTIME, 0, &ts, NULL, FUTEX_BITSET_MATCH_ANY) ? errno : 0; }
    if (!strcmp(a, "sem_clockwait_rt")) { ts = clk(CLOCK_REALTIME, s); return sem_clockwait(&sem, CLOCK_REALTIME, &ts) ? errno : 0; }
    if (!strcmp(a, "mutex_clocklock_rt")) { hold(); ts = clk(CLOCK_REALTIME, s); return pthread_mutex_clocklock(&held, CLOCK_REALTIME, &ts); }
    if (!strcmp(a, "cond_timedwait_gtod")) { struct timeval tv; gettimeofday(&tv, NULL); ts.tv_sec = tv.tv_sec; ts.tv_nsec = tv.tv_usec * 1000; ts = plus(ts, s); pthread_mutex_lock(&m); return pthread_cond_timedwait(&c_rt, &m, &ts); }
    if (!strcmp(a, "gtod_poll_loop")) {
        /* a deadline from gettimeofday, polled with relative sleeps of at most 100 ms */
        struct timeval tv; gettimeofday(&tv, NULL); double end = tv.tv_sec + tv.tv_usec / 1e6 + s;
        for (;;) { gettimeofday(&tv, NULL); double left = end - (tv.tv_sec + tv.tv_usec / 1e6); if (left <= 0) return 0; usleep((useconds_t)((left < 0.1 ? left : 0.1) * 1e6)); }
    }
    fprintf(stderr, "unknown %s\n", a);
    return -1;
}

int main(int argc, char **argv) {
    pthread_condattr_t at; pthread_condattr_init(&at); pthread_condattr_setclock(&at, CLOCK_MONOTONIC); pthread_cond_init(&c_mono, &at);
    sem_init(&sem, 0, 0); sem_init(&tsem, 0, 0);
    double s = argc > 2 ? atof(argv[2]) : 1.0;
    double r0 = secs(CLOCK_MONOTONIC_RAW), u0 = secs(CLOCK_REALTIME), m0 = secs(CLOCK_MONOTONIC);
    long r = run(argv[1], s);
    double r1 = secs(CLOCK_MONOTONIC_RAW), u1 = secs(CLOCK_REALTIME), m1 = secs(CLOCK_MONOTONIC);
    if (r == ETIMEDOUT) r = 0;
    printf("{\"api\": \"%s\", \"lang\": \"c\", \"real\": %.6f, \"u_rt\": %.6f, \"u_mono\": %.6f, \"rc\": %ld, \"start_raw\": %.6f}\n",
           argv[1], r1 - r0, u1 - u0, m1 - m0, r, r0);
    return 0;
}
