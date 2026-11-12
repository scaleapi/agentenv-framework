/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * agent-env's additions to libfaketime 0.9.12, part 2, included by libfaketime.c (libfaketime.patch) after its time
 * functions: clock reads and waits for a set realtime clock time on universe time, re-planned whenever the page
 * changes, and everything else (monotonic and boot clocks, relative sleeps and timeouts) the kernel's.
 * Derived from libfaketime, so licensed as it is, under the GNU General Public License version 2 (COPYING).
 */
/*
 * Elapsed time is real; clock reads and waits for a realtime clock time follow the universe.
 */
#include <sys/syscall.h>
#include <linux/futex.h>
#include <sched.h>
#include <stdarg.h>

#if !defined(FAKE_PTHREAD) || !defined(FAKE_TIMERS) || !defined(FAKE_SLEEP) || defined(INTERCEPT_SYSCALL)
#error agent-env waits need FAKE_PTHREAD, FAKE_TIMERS and FAKE_SLEEP, and their own syscall(), not the INTERCEPT_SYSCALL one
#endif
#ifndef SYS_futex_waitv
#define SYS_futex_waitv 449
#endif

/*
 * the system call itself, with no libc and no libfaketime state, so syscall() below works before the
 * library is initialised (Chromium's allocator calls syscall() from inside the dynamic loader).
 */
static long ft_raw_syscall6(long n, long a, long b, long c, long d, long e, long f)
{
  long ret;
#if defined(__x86_64__)
  register long r10 __asm__("r10") = d;
  register long r8 __asm__("r8") = e;
  register long r9 __asm__("r9") = f;
  __asm__ volatile("syscall" : "=a"(ret) : "0"(n), "D"(a), "S"(b), "d"(c), "r"(r10), "r"(r8), "r"(r9)
                   : "rcx", "r11", "memory");
#elif defined(__aarch64__)
  register long x8 __asm__("x8") = n;
  register long x0 __asm__("x0") = a;
  register long x1 __asm__("x1") = b;
  register long x2 __asm__("x2") = c;
  register long x3 __asm__("x3") = d;
  register long x4 __asm__("x4") = e;
  register long x5 __asm__("x5") = f;
  __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8), "r"(x1), "r"(x2), "r"(x3), "r"(x4), "r"(x5) : "memory");
  ret = x0;
#else
#error the agent-env syscall() needs x86_64 or aarch64
#endif
  if (ret < 0 && ret > -4096)
  {
    errno = (int) -ret;
    return -1;
  }
  return ret;
}

#define FT_NS 1000000000LL
#define FT_NEVER INT64_MAX
static __thread struct ft_clock_page ft_tl_snap;
static __thread bool ft_tl_snap_ok = false;

/* a consistent copy of the page. While the follower holds it odd (preempted mid-write), wait for it on seq,
   which it wakes when done, up to a second: a copy older than the page being written would be extrapolated past the
   new page's anchor, and the clock would step back when the write lands. Only a follower that died mid-write (seq odd
   for over a second, remembered) leaves this thread on its last good copy. */
static uint64_t ft_dead_seq = 0;
static bool ft_snap(struct ft_clock_page *s)
{
  for (int i = 0; i < 64; i++)
  {
    if (ft_page_snapshot(s))
    {
      ft_tl_snap = *s;
      ft_tl_snap_ok = true;
      return true;
    }
    sched_yield();
  }
  for (int i = 0; i < 100; i++)
  {
    uint64_t seq = __atomic_load_n(&ft_page->seq, __ATOMIC_ACQUIRE);
    if ((seq & 1) && seq == __atomic_load_n(&ft_dead_seq, __ATOMIC_RELAXED)) break;
    if (seq & 1)
    {
      struct timespec ts = {0, 10000000};
      ft_raw_syscall6(SYS_futex, (long) &ft_page->seq, FUTEX_WAIT, (long) (uint32_t) seq, (long) &ts, 0, 0);
    }
    if (ft_page_snapshot(s))
    {
      ft_tl_snap = *s;
      ft_tl_snap_ok = true;
      return true;
    }
    if (i == 99) __atomic_store_n(&ft_dead_seq, seq, __ATOMIC_RELAXED);
  }
  if (!ft_tl_snap_ok) return false;
  *s = ft_tl_snap;
  return true;
}

/* the real time just read, in hand before the loads after it: a load can be carried out before an earlier read of the
   CPU's counter (aarch64 and x86_64 both allow it). On aarch64 a load whose address depends on the time, then a load
   barrier, as the kernel orders its own counter reads; on x86_64 an lfence. */
static inline void ft_after_real_read(const struct timespec *tp)
{
#if defined(__aarch64__)
  uint64_t tmp;
  __asm__ volatile("eor %0, %1, %1\n\tadd %0, sp, %0\n\tldr xzr, [%0]\n\tdmb ishld"
                   : "=&r"(tmp) : "r"((uint64_t) tp->tv_nsec) : "memory");
#elif defined(__x86_64__)
  (void) tp;
  __asm__ volatile("lfence" ::: "memory");
#else
  (void) tp;
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/* a consistent copy of the page and the real time read while it was the page: the time is read after the copy and kept
   only if the page is still the one copied. A time from before a write paired with the page it wrote (a reader that
   waited out the write, or was descheduled between its time and its copy) could put the universe behind what that
   reader had shown, since a write is continuous with the page it replaces only at its own real time. */
static bool ft_snap_at(clockid_t clock, struct ft_clock_page *s, struct timespec *tp)
{
  for (int i = 0; i < 8; i++)
  {
    int r;
    if (!ft_snap(s)) return false;
    DONT_FAKE_TIME(r = (*real_clock_gettime)(clock, tp));
    if (r != 0) return false;
    ft_after_real_read(tp);
    if (__atomic_load_n(&ft_page->seq, __ATOMIC_ACQUIRE) == s->seq) return true;
  }
  return true;
}

static inline int64_t ft_ns(const struct timespec *t)
{
  if (t->tv_sec > 9000000000LL) return FT_NEVER / 2;
  if (t->tv_sec < -9000000000LL) return -(FT_NEVER / 2);
  return (int64_t) t->tv_sec * FT_NS + t->tv_nsec;
}
static inline struct timespec ft_ts(int64_t ns)
{
  struct timespec t;
  if (ns < 0) ns = 0;
  t.tv_sec = (time_t) (ns / FT_NS);
  t.tv_nsec = (long) (ns % FT_NS);
  return t;
}
static inline int64_t ft_real_rt(void)
{
  struct timespec t;
  (*real_clock_gettime)(CLOCK_REALTIME, &t);
  return ft_ns(&t);
}
static inline bool ft_uw_active(void)
{
  return ft_page != NULL && !ft_disabled && !dont_fake;
}
static inline bool ft_faked(void)
{
  return !ft_disabled && !dont_fake;
}
static inline bool ft_valid_ts(const struct timespec *t)
{
  return t != NULL && t->tv_nsec >= 0 && t->tv_nsec < FT_NS && t->tv_sec >= 0;
}
static inline bool ft_rt_clock(clockid_t c)
{
  return c == CLOCK_REALTIME || c == CLOCK_REALTIME_ALARM;
}
/* the clocks the universe has; every other one (monotonic, boot, raw, CPU time) is real. */
static inline bool ft_universe_clock(clockid_t c)
{
  return c == CLOCK_REALTIME || c == CLOCK_REALTIME_COARSE || c == CLOCK_REALTIME_ALARM || c == CLOCK_TAI;
}

/* the universe's realtime at real realtime t; before the anchor, the anchor's. */
static int64_t ft_page_rt(const struct ft_clock_page *s, int64_t t)
{
  if (!s->armed) return t;
  if (s->rate == 0 || t <= s->anchor_ns) return s->universe_ns;
  if (s->rate == 1.0) return s->universe_ns + (t - s->anchor_ns);
  return s->universe_ns + (int64_t) ((double) (t - s->anchor_ns) * s->rate);
}
/* the real realtime at which the universe reaches ``due``; 0 (the past) once it has, FT_NEVER while frozen
   short of it. */
static int64_t ft_real_of(const struct ft_clock_page *s, int64_t due)
{
  double r;
  if (!s->armed) return due < 1 ? 1 : due;
  if (due <= s->universe_ns) return 1;
  if (s->rate <= 0) return FT_NEVER;
  r = (double) (due - s->universe_ns) / s->rate;
  if (r > 4e18) return FT_NEVER;
  return s->anchor_ns + (int64_t) r + 1;
}
static inline bool ft_reached(const struct ft_clock_page *s, int64_t due)
{
  return ft_page_rt(s, ft_real_rt()) >= due;
}
static inline uint32_t ft_seq32(void)
{
  return (uint32_t) __atomic_load_n(&ft_page->seq, __ATOMIC_ACQUIRE);
}

/* ---- a lock for the tables below, with signals blocked so a handler can't deadlock on it ---- */
static volatile int ft_tab_lock_word = 0;
static void ft_tab_lock(sigset_t *old)
{
  sigset_t all;
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, old);
  while (__atomic_exchange_n(&ft_tab_lock_word, 1, __ATOMIC_ACQUIRE)) sched_yield();
}
static void ft_tab_unlock(const sigset_t *old)
{
  __atomic_store_n(&ft_tab_lock_word, 0, __ATOMIC_RELEASE);
  pthread_sigmask(SIG_SETMASK, old, NULL);
}

/*
 * ---- timers for a realtime clock time: timerfds and POSIX timers armed at TIMER_ABSTIME on a realtime clock. The
 * kernel timer is armed at the real realtime the universe reaches the deadline at (in the past once it has, so it fires
 * now), its interval at interval / rate, and disarmed while frozen short of it; the helper arms it again on each change.
 * A one-shot timer that has fired is left alone. ----
 */
#define FT_TIMER_FDS 8
struct ft_timer
{
  int kind;                      /* 0 free, 1 timerfd, 2 POSIX timer */
  int tracked;                   /* armed for a realtime clock time */
  int nfds, fds[FT_TIMER_FDS];   /* a timerfd's descriptors: the one it was armed through and its duplicates */
  timer_t id;
  int clock;
  int flags;                     /* the caller's flags, TFD_TIMER_CANCEL_ON_SET kept */
  int real_set;                  /* the kernel timer is armed */
  int64_t due, interval;         /* universe ns */
  struct ft_clock_page pg;       /* the page it was last armed from */
};
#define FT_TIMERS_MAX 1024
static struct ft_timer ft_timers[FT_TIMERS_MAX];
static int ft_timers_top = 0;
static volatile int ft_timers_any = 0;
static int (*real_timer_create_new)(clockid_t, struct sigevent *, timer_t *);
static int (*real_timer_delete_new)(timer_t);
static int (*real_timer_settime_new)(timer_t, int, const struct itimerspec *, struct itimerspec *);
static int (*real_timer_gettime_new)(timer_t, struct itimerspec *);
#define FT_TFD_CLOCKS 65536
static int ft_tfd_clock[FT_TFD_CLOCKS];   /* clock + 1 as timerfd_create was told; 0 unknown */

static struct ft_timer *ft_timer_find_fd(int fd)
{
  for (int i = 0; i < ft_timers_top; i++)
    if (ft_timers[i].kind == 1)
      for (int k = 0; k < ft_timers[i].nfds; k++)
        if (ft_timers[i].fds[k] == fd) return &ft_timers[i];
  return NULL;
}
static struct ft_timer *ft_timer_find_id(timer_t id)
{
  for (int i = 0; i < ft_timers_top; i++)
    if (ft_timers[i].kind == 2 && ft_timers[i].id == id) return &ft_timers[i];
  return NULL;
}
static struct ft_timer *ft_timer_new(void)
{
  for (int i = 0; i < FT_TIMERS_MAX; i++)
    if (ft_timers[i].kind == 0)
    {
      memset(&ft_timers[i], 0, sizeof ft_timers[i]);
      if (i >= ft_timers_top) ft_timers_top = i + 1;
      return &ft_timers[i];
    }
  return NULL;
}
static int ft_timer_real_get(struct ft_timer *t, struct itimerspec *cur)
{
  return t->kind == 1 ? (*real_timerfd_gettime)(t->fds[0], cur) : (*real_timer_gettime_new)(t->id, cur);
}
/* the next expiration at or after ``after``. */
static int64_t ft_timer_next(const struct ft_timer *t, int64_t after)
{
  int64_t next = t->due;
  if (t->interval > 0 && next < after) next += ((after - next + t->interval - 1) / t->interval) * t->interval;
  return next;
}
/* the kernel timer set from page ``s``. A periodic timer goes on from its first expiration past the
   universe's time on the page it was armed from, so a jump delivers what it passed at once. Lock held. */
static int ft_timer_arm(struct ft_timer *t, const struct ft_clock_page *s, bool first)
{
  struct itimerspec real, cur;
  double rate = s->armed ? s->rate : 1.0;
  int64_t next = t->due, at;
  int r;
  memset(&real, 0, sizeof real);
  if (!first && t->real_set && t->interval == 0 && ft_timer_real_get(t, &cur) == 0 && cur.it_value.tv_sec == 0
      && cur.it_value.tv_nsec == 0)
  {
    t->tracked = 0;   /* fired: the expiration is the kernel's to hand over */
    return 0;
  }
  if (!first && t->interval > 0) next = ft_timer_next(t, ft_page_rt(&t->pg, ft_real_rt()) + 1);
  at = ft_real_of(s, next);
  if (at != FT_NEVER)
  {
    real.it_value = ft_ts(at);
    if (real.it_value.tv_sec == 0 && real.it_value.tv_nsec == 0) real.it_value.tv_nsec = 1;
    if (t->interval > 0)
    {
      int64_t iv = (int64_t) ((double) t->interval / rate);
      real.it_interval = ft_ts(iv < 1 ? 1 : iv);
    }
  }
  t->due = next;
  t->pg = *s;
  t->real_set = (at != FT_NEVER);
  if (t->kind == 1)
    r = (*real_timerfd_settime)(t->fds[0], TFD_TIMER_ABSTIME | (t->flags & TFD_TIMER_CANCEL_ON_SET), &real, NULL);
  else
    r = (*real_timer_settime_new)(t->id, TIMER_ABSTIME, &real, NULL);
  if (r != 0) t->tracked = 0;
  return r;
}
/* the timer's value in universe time. Lock held. */
static void ft_timer_value(struct ft_timer *t, struct itimerspec *out)
{
  struct ft_clock_page s;
  struct itimerspec cur;
  memset(out, 0, sizeof *out);
  if (!t->tracked) return;
  if (t->interval == 0 && t->real_set && ft_timer_real_get(t, &cur) == 0 && cur.it_value.tv_sec == 0
      && cur.it_value.tv_nsec == 0)
  {
    t->tracked = 0;
    return;
  }
  out->it_interval = ft_ts(t->interval);
  if (ft_snap(&s))
  {
    int64_t now = ft_page_rt(&s, ft_real_rt()), next = ft_timer_next(t, now);
    out->it_value = ft_ts(next - now < 1 ? 1 : next - now);
  }
}

/*
 * ---- realtime condvar waits in progress: each waits to the real time its deadline falls at; on a page change that
 * moves that time, the helper broadcasts its condvar, once, and the wait returns 0 (a spurious wakeup the caller
 * rechecks), so the caller waits again on the new page. A broadcast that comes before the waiter is inside glibc's wait
 * is sent again until the waiter has seen the change. ----
 */
struct ft_cw
{
  pthread_cond_t *cond;
  int used, woken;
  int64_t due, real_due;
  uint64_t gen;            /* the page generation its real_due came from; FT_CW_UNPLANNED before */
};
#define FT_CW_MAX 512
#define FT_CW_UNPLANNED UINT64_MAX
static struct ft_cw ft_cws[FT_CW_MAX];
static int ft_cws_top = 0;

/* ---- the helper thread: one per process, started with the first such timer or wait, blocked on the page's seq ---- */
static volatile pid_t ft_helper_pid = 0;
static volatile int ft_atfork_set = 0;
static void ft_helper_child(void)
{
  /* POSIX timers and condvar waiters belong to the parent; timerfds are shared with it and stay */
  for (int i = 0; i < ft_timers_top; i++)
    if (ft_timers[i].kind == 2) ft_timers[i].kind = 0;
  memset(ft_cws, 0, sizeof ft_cws);
  ft_cws_top = 0;
  ft_tab_lock_word = 0;
  ft_helper_pid = 0;
}

/* on a page change, every timer armed again and every condvar waiter whose real deadline moved woken; true
   while a waiter is still to see it (not planned yet, or broadcast to before it was waiting). */
static bool ft_on_change(const struct ft_clock_page *s, bool rebroadcast)
{
  sigset_t old;
  bool pending = false;
  ft_tab_lock(&old);
  if (!rebroadcast)
    for (int i = 0; i < ft_timers_top; i++)
    {
      struct ft_timer *t = &ft_timers[i];
      if (t->kind != 0 && t->tracked && t->pg.generation != s->generation) ft_timer_arm(t, s, false);
    }
  for (int i = 0; i < ft_cws_top; i++)
  {
    struct ft_cw *w = &ft_cws[i];
    if (!w->used || w->gen == s->generation) continue;
    if (w->gen == FT_CW_UNPLANNED)
    {
      pending = true;
      continue;
    }
    if (!w->woken)
    {
      int64_t nd = ft_real_of(s, w->due), diff;
      diff = (nd == FT_NEVER || w->real_due == FT_NEVER) ? (nd == w->real_due ? 0 : FT_NEVER) : nd - w->real_due;
      if (diff >= -1000000 && diff <= 1000000)
      {
        w->gen = s->generation;   /* its real end didn't move */
        continue;
      }
    }
    if (!w->woken || rebroadcast) pthread_cond_broadcast(w->cond);
    w->woken = 1;
    pending = true;
  }
  ft_tab_unlock(&old);
  return pending;
}

static void *ft_helper_main(void *arg)
{
  struct ft_clock_page s;
  uint64_t gen = UINT64_MAX;   /* the first pass catches a change made while it started */
  int backoff_ms = 0;
  (void) arg;
  for (;;)
  {
    uint32_t seq32 = ft_seq32();
    bool pending = false;
    if (ft_snap(&s))
    {
      if (s.generation != gen)
      {
        gen = s.generation;
        pending = ft_on_change(&s, false);
        backoff_ms = pending ? 1 : 0;
      }
      else if (backoff_ms > 0)
      {
        pending = ft_on_change(&s, true);
        backoff_ms = pending ? (backoff_ms < 64 ? backoff_ms * 2 : 100) : 0;
      }
    }
    if (backoff_ms > 0)
    {
      struct timespec ts = {0, (long) backoff_ms * 1000000L};
      ft_raw_syscall6(SYS_futex, (long) &ft_page->seq, FUTEX_WAIT, (long) seq32, (long) &ts, 0, 0);
    }
    else
      ft_raw_syscall6(SYS_futex, (long) &ft_page->seq, FUTEX_WAIT, (long) seq32, 0, 0, 0);
  }
  return NULL;
}

static void ft_helper_start(void)
{
  pid_t pid = getpid();
  sigset_t all, old;
  pthread_t thread;
  pthread_attr_t attr;
  if (__atomic_load_n(&ft_helper_pid, __ATOMIC_ACQUIRE) == pid) return;
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, &old);
  while (__atomic_exchange_n(&ft_tab_lock_word, 1, __ATOMIC_ACQUIRE)) sched_yield();
  if (ft_helper_pid != pid)
  {
    if (!ft_atfork_set)
    {
      pthread_atfork(NULL, NULL, ft_helper_child);
      ft_atfork_set = 1;
    }
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setstacksize(&attr, 128 * 1024);
    if (pthread_create(&thread, &attr, ft_helper_main, NULL) == 0)  /* inherits the all-blocked signal mask */
    {
      __atomic_store_n(&ft_helper_pid, pid, __ATOMIC_RELEASE);
      pthread_setname_np(thread, "agentenv-clock");
    }
    pthread_attr_destroy(&attr);
  }
  __atomic_store_n(&ft_tab_lock_word, 0, __ATOMIC_RELEASE);
  pthread_sigmask(SIG_SETMASK, &old, NULL);
}

/* ---- the waits ---- */

/* the raw system calls below as the cancellation points their callers are (clock_nanosleep, sem_timedwait):
   cancellation acted on at once for the call alone, as glibc's own waits were before its cancellable system call
   wrapper. */
static inline int ft_async_cancel(void)
{
  int old;
  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &old);
  return old;
}
static inline void ft_deferred_cancel(int old)
{
  int e = errno, ignored;
  pthread_setcanceltype(old, &ignored);
  errno = e;
}

/* a sleep until the universe reaches ``due``, on the page's seq. 0, or EINTR. */
static int ft_rt_sleep(const struct ft_clock_page *first, int64_t due)
{
  struct ft_clock_page s = *first;
  pthread_testcancel();
  for (;;)
  {
    uint32_t seq32 = (uint32_t) s.seq;
    int64_t until;
    struct timespec ts;
    long r;
    int ct;
    if (ft_reached(&s, due)) return 0;
    until = ft_real_of(&s, due);
    ts = ft_ts(until);
    ct = ft_async_cancel();
    r = ft_raw_syscall6(SYS_futex, (long) &ft_page->seq, FUTEX_WAIT_BITSET | FUTEX_CLOCK_REALTIME, (long) seq32,
                        until == FT_NEVER ? 0 : (long) &ts, 0, FUTEX_BITSET_MATCH_ANY);
    ft_deferred_cancel(ct);
    if (r == -1 && errno == EINTR) return EINTR;
    if (!ft_snap(&s)) return 0;
  }
}

struct ft_futex_waitv { uint64_t val, uaddr; uint32_t flags, reserved; };
static volatile int ft_waitv_missing = 0;
enum { FT_W_WORD, FT_W_PAGE, FT_W_TIMEOUT, FT_W_INTR, FT_W_ERR };
/* without futex_waitv (before Linux 5.16, under gVisor, or refused by a seccomp filter): the word alone, until the real
   deadline or a slice (AGENTENV_WAIT_SLICE_MS) at most, and the caller reads the page again between slices. */
static int ft_wait1(uint32_t *word, uint32_t val, bool private_word, int64_t until)
{
  int64_t end = ft_real_rt() + ft_wait_slice_ns;
  struct timespec ts;
  if (until < end) end = until;
  ts = ft_ts(end);
  if (ft_raw_syscall6(SYS_futex, (long) word,
                      FUTEX_WAIT_BITSET | FUTEX_CLOCK_REALTIME | (private_word ? FUTEX_PRIVATE_FLAG : 0), (long) val,
                      (long) &ts, 0, FUTEX_BITSET_MATCH_ANY) == 0 || errno == EAGAIN)
    return FT_W_WORD;
  if (errno == ETIMEDOUT) return FT_W_PAGE;   /* a slice's end, or the deadline: the caller reads the page */
  return errno == EINTR ? FT_W_INTR : FT_W_ERR;
}
/* until *word isn't val, the page changes from seq32, or real realtime ``until``; FT_W_ERR with errno the kernel's
   (an unaligned or unmapped word). */
static int ft_wait2(uint32_t *word, uint32_t val, bool private_word, uint32_t seq32, int64_t until)
{
  struct ft_futex_waitv w[2];
  struct timespec ts = ft_ts(until);
  long r;
  if (ft_waitv_missing) return ft_wait1(word, val, private_word, until);
  /* the page first: futex_waitv returns the last of the futexes woken, so a caller's wake is never taken for the page's */
  w[0].val = seq32;
  w[0].uaddr = (uint64_t) (uintptr_t) &ft_page->seq;
  w[0].flags = 0x02; /* FUTEX2_SIZE_U32, shared */
  w[0].reserved = 0;
  w[1].val = val;
  w[1].uaddr = (uint64_t) (uintptr_t) word;
  w[1].flags = 0x02 | (private_word ? 128 : 0);
  w[1].reserved = 0;
  r = ft_raw_syscall6(SYS_futex_waitv, (long) w, 2, 0, until == FT_NEVER ? 0 : (long) &ts, CLOCK_REALTIME, 0);
  if (r == 1) return FT_W_WORD;
  if (r == 0) return FT_W_PAGE;
  if (errno == EAGAIN) return __atomic_load_n(word, __ATOMIC_RELAXED) != val ? FT_W_WORD : FT_W_PAGE;
  if (errno == ETIMEDOUT) return FT_W_TIMEOUT;
  if (errno == EINTR) return FT_W_INTR;
  if (errno != ENOSYS && errno != EPERM) return FT_W_ERR;   /* EPERM: a seccomp filter that refuses it so */
  ft_waitv_missing = 1;
  return ft_wait1(word, val, private_word, until);
}

/* futex(2) FUTEX_WAIT_BITSET|FUTEX_CLOCK_REALTIME through syscall(): until the universe reaches the deadline,
   beside the page; *handled false leaves the call to the kernel as it is. */
static long ft_u_futex_rt(uint32_t *uaddr, int op, uint32_t val, const struct timespec *abs, bool *handled)
{
  struct ft_clock_page s;
  int64_t due = ft_ns(abs);
  *handled = false;
  if (!ft_snap(&s)) return 0;
  *handled = true;
  for (;;)
  {
    uint32_t seq32 = (uint32_t) s.seq;
    int k;
    if (ft_reached(&s, due))
    {
      errno = ETIMEDOUT;
      return -1;
    }
    k = ft_wait2(uaddr, val, (op & FUTEX_PRIVATE_FLAG) != 0, seq32, ft_real_of(&s, due));
    if (k == FT_W_WORD)
    {
      if (__atomic_load_n(uaddr, __ATOMIC_RELAXED) != val)
      {
        errno = EAGAIN;
        return -1;
      }
      return 0;
    }
    if (k == FT_W_INTR)
    {
      errno = EINTR;
      return -1;
    }
    if (k == FT_W_ERR) return -1;
    if (!ft_snap(&s)) return 0;
  }
}

/* sem_timedwait / sem_clockwait(CLOCK_REALTIME) on glibc's semaphore (its layout is part of the ABI, since
   process-shared semaphores outlive glibc versions): the value in the low word of data, the waiters above it, the
   futex the value word, which sem_post wakes when there are waiters; ``private`` is glibc's futex flag the other way
   round (0 for a process-private semaphore, FUTEX_PRIVATE_FLAG for a shared one). */
struct ft_new_sem { uint64_t data; int private_flag; int pad; };
static void ft_sem_unwait(void *sem)
{
  __atomic_fetch_sub(&((struct ft_new_sem *) sem)->data, (uint64_t) 1 << 32, __ATOMIC_RELAXED);
}
/* as one of the semaphore's waiters (counted in data's high word): 0 with a token taken, or the error. A cancelled wait
   leaves the waiters it joined, as glibc's does. */
static int ft_sem_wait_slow(struct ft_new_sem *sem, struct ft_clock_page *s, int64_t due)
{
  volatile int err = 0;
  pthread_cleanup_push(ft_sem_unwait, sem);
  uint64_t d = __atomic_fetch_add(&sem->data, (uint64_t) 1 << 32, __ATOMIC_RELAXED);
  for (;;)
  {
    if ((d & 0xffffffffu) == 0)
    {
      int k, ct;
      if (ft_reached(s, due))
      {
        err = ETIMEDOUT;
        break;
      }
      ct = ft_async_cancel();
      k = ft_wait2((uint32_t *) &sem->data, 0, sem->private_flag == 0, (uint32_t) s->seq, ft_real_of(s, due));
      ft_deferred_cancel(ct);
      if (k == FT_W_INTR || k == FT_W_ERR)
      {
        err = k == FT_W_INTR ? EINTR : errno;
        break;
      }
      if (!ft_snap(s))
      {
        err = ETIMEDOUT;
        break;
      }
      d = __atomic_load_n(&sem->data, __ATOMIC_RELAXED);
    }
    else if (__atomic_compare_exchange_n(&sem->data, &d, d - 1 - ((uint64_t) 1 << 32), true, __ATOMIC_ACQUIRE,
                                         __ATOMIC_RELAXED))
      break;
  }
  pthread_cleanup_pop(err != 0);
  return err;
}
static int ft_u_sem_wait(sem_t *sem_, int64_t due, bool *handled)
{
  struct ft_new_sem *sem = (struct ft_new_sem *) sem_;
  struct ft_clock_page s;
  uint64_t d;
  int err;
  *handled = false;
  if (sizeof(sem_t) < sizeof(struct ft_new_sem) || !ft_snap(&s)) return 0;
  *handled = true;
  pthread_testcancel();
  d = __atomic_load_n(&sem->data, __ATOMIC_RELAXED);
  while ((d & 0xffffffffu) > 0)
    if (__atomic_compare_exchange_n(&sem->data, &d, d - 1, true, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return 0;
  err = ft_sem_wait_slow(sem, &s, due);
  if (err == 0) return 0;
  errno = err;
  return -1;
}

static void ft_cw_release(void *w)
{
  sigset_t old;
  ft_tab_lock(&old);
  ((struct ft_cw *) w)->used = 0;
  ft_tab_unlock(&old);
}
/* glibc's wait, a cancellation point: a cancelled wait gives up its entry, so the helper never wakes a condvar that
   may be gone. */
static int ft_cw_clockwait(struct ft_cw *w, pthread_cond_t *cond, pthread_mutex_t *mutex, const struct timespec *end)
{
  volatile int r;
  pthread_cleanup_push(ft_cw_release, w);
  r = real_pthread_cond_clockwait(cond, mutex, CLOCK_REALTIME, end);
  pthread_cleanup_pop(0);
  return r;
}

/* a realtime condvar wait (see ft_cw). -1 leaves it to the caller's own path (no page, table full). */
static int ft_u_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex, const struct timespec *abstime)
{
  struct ft_clock_page s;
  struct ft_cw *w = NULL;
  struct timespec end;
  sigset_t old;
  int64_t due = ft_ns(abstime), real_due;
  int r, woken;
  if (!ft_snap(&s)) return -1;
  if (ft_reached(&s, due)) return ETIMEDOUT;
  ft_helper_start();
  ft_tab_lock(&old);
  for (int i = 0; i < FT_CW_MAX; i++)
    if (!ft_cws[i].used)
    {
      w = &ft_cws[i];
      if (i >= ft_cws_top) ft_cws_top = i + 1;
      break;
    }
  if (w != NULL)
  {
    w->cond = cond;
    w->due = due;
    w->woken = 0;
    w->gen = FT_CW_UNPLANNED;
    w->used = 1;
  }
  ft_tab_unlock(&old);
  if (w == NULL) return -1;
  if (!ft_snap(&s)) s.armed = 0;
  real_due = ft_real_of(&s, due);
  ft_tab_lock(&old);
  w->real_due = real_due;
  w->gen = s.generation;
  ft_tab_unlock(&old);
  end = ft_ts(real_due == FT_NEVER ? ft_real_rt() + 86400 * FT_NS : real_due);
  r = ft_cw_clockwait(w, cond, mutex, &end);
  ft_tab_lock(&old);
  woken = w->woken;
  w->used = 0;
  ft_tab_unlock(&old);
  /* a real timeout before the universe's deadline (the page moved it later) is a spurious wakeup; the helper's wake for
     a jump that passed the deadline is the timeout */
  if (r == ETIMEDOUT) return (ft_snap(&s) && !ft_reached(&s, due)) ? 0 : ETIMEDOUT;
  if (r == 0 && woken && ft_snap(&s) && ft_reached(&s, due)) return ETIMEDOUT;
  return r;
}

/* a realtime mutex wait: glibc's mutex has no way in, so slices (AGENTENV_WAIT_SLICE_MS) of a real
   CLOCK_MONOTONIC wait, which loses nothing when it times out, with the page read between them. */
static int ft_u_mutex_lock(pthread_mutex_t *mutex, int64_t due, bool *handled)
{
  struct ft_clock_page s;
  *handled = false;
  if (!ft_snap(&s)) return 0;
  *handled = true;
  for (;;)
  {
    struct timespec now, end;
    int64_t at, sl;
    int r;
    if (ft_reached(&s, due)) return ETIMEDOUT;
    at = ft_real_of(&s, due);
    sl = (at == FT_NEVER) ? ft_wait_slice_ns : at - ft_real_rt();
    if (sl > ft_wait_slice_ns) sl = ft_wait_slice_ns;
    if (sl < 1) sl = 1;
    (*real_clock_gettime)(CLOCK_MONOTONIC, &now);
    end = ft_ts(ft_ns(&now) + sl);
    r = real_pthread_mutex_clocklock(mutex, CLOCK_MONOTONIC, &end);
    if (r == EINVAL) /* a PI mutex on a kernel without FUTEX_LOCK_PI2 */
    {
      struct timespec real = ft_ts(ft_real_rt() + sl);
      r = real_pthread_mutex_timedlock(mutex, &real);
    }
    if (r != ETIMEDOUT) return r;
    if (!ft_snap(&s)) return ETIMEDOUT;
  }
}

/* syscall(): futex waits for a realtime clock time follow the universe (raw futex users such as Rust's std
   and GLib call it), clock_gettime reads it once the library is up; everything else goes straight to the kernel. */
long syscall(long number, ...)
{
  va_list ap;
  long a[6];
  va_start(ap, number);
  for (int i = 0; i < 6; i++) a[i] = va_arg(ap, long);
  va_end(ap);
  if (number == SYS_futex && ft_uw_active())
  {
    int op = (int) a[1];
    const struct timespec *ts = (const struct timespec *) a[3];
    if ((op & FUTEX_CMD_MASK) == FUTEX_WAIT_BITSET && (op & FUTEX_CLOCK_REALTIME) && ft_valid_ts(ts)
        && (uint32_t) a[5] == FUTEX_BITSET_MATCH_ANY)
    {
      bool handled;
      long r = ft_u_futex_rt((uint32_t *) a[0], op, (uint32_t) a[2], ts, &handled);
      if (handled) return r;
    }
  }
  else if (number == SYS_clock_gettime && ft_page != NULL && !ft_disabled && !dont_fake)
    return clock_gettime((clockid_t) a[0], (struct timespec *) a[1]);
  return ft_raw_syscall6(number, a[0], a[1], a[2], a[3], a[4], a[5]);
}

/* ---- timerfds ---- */
static int ft_u_timerfd_settime(int fd, int flags, const struct itimerspec *nv, struct itimerspec *ov, bool *handled)
{
  struct ft_clock_page s;
  struct ft_timer *t = NULL;
  struct itimerspec prev;
  sigset_t old;
  int clock, r;
  bool was = false;
  *handled = false;
  if (fd < 0 || fd >= FT_TFD_CLOCKS || nv == NULL) return 0;
  clock = ft_tfd_clock[fd] - 1;
  *handled = true;
  if (ft_timers_any)
  {
    ft_tab_lock(&old);
    t = ft_timer_find_fd(fd);
    if (t != NULL && t->tracked)
    {
      ft_timer_value(t, &prev);
      was = true;
      t->tracked = 0;   /* the record stays, with the timer's descriptors, while any is open */
    }
    ft_tab_unlock(&old);
  }
  if (!ft_rt_clock(clock) || !(flags & TFD_TIMER_ABSTIME) || !ft_valid_ts(&nv->it_value)
      || !ft_valid_ts(&nv->it_interval) || (nv->it_value.tv_sec == 0 && nv->it_value.tv_nsec == 0) || !ft_snap(&s))
  {
    r = (*real_timerfd_settime)(fd, flags, nv, ov);
    if (r == 0 && ov != NULL && was) *ov = prev;
    return r;
  }
  ft_tab_lock(&old);
  if ((t = ft_timer_find_fd(fd)) == NULL && (t = ft_timer_new()) != NULL)
  {
    t->kind = 1;
    t->nfds = 1;
    t->fds[0] = fd;
  }
  if (t == NULL)
  {
    ft_tab_unlock(&old);
    *handled = false;   /* table full: the caller's own (translated once) path */
    return 0;
  }
  t->clock = clock;
  t->flags = flags;
  t->due = ft_ns(&nv->it_value);
  t->interval = ft_ns(&nv->it_interval);
  t->tracked = 1;
  if (ov != NULL)
  {
    if (was)
      *ov = prev;
    else
      (*real_timerfd_gettime)(fd, ov);
  }
  r = ft_timer_arm(t, &s, true);
  if (r != 0) t->tracked = 0;
  ft_timers_any = 1;
  ft_tab_unlock(&old);
  if (r == 0) ft_helper_start();
  return r;
}
static int ft_u_timerfd_gettime(int fd, struct itimerspec *cur, bool *handled)
{
  struct ft_timer *t;
  sigset_t old;
  *handled = false;
  if (!ft_timers_any || cur == NULL) return 0;
  ft_tab_lock(&old);
  t = ft_timer_find_fd(fd);
  if (t != NULL && t->tracked)
  {
    ft_timer_value(t, cur);
    *handled = true;
  }
  ft_tab_unlock(&old);
  return 0;
}

/*
 * ---- descriptors: a timerfd's record follows its file, not a number. close() takes a descriptor off it (the record
 * goes with the last), dup() and its kin add the new one, so the helper never arms a number the program has closed or
 * that now names another file. A number closed where this can't see (closefrom, a raw system call) is dropped when
 * timerfd_create hands it out again; a timer with more duplicates than a record holds is left armed as it was. ----
 */
static int (*ft_real_close)(int);
static int (*ft_real_dup)(int);
static int (*ft_real_dup2)(int, int);
static int (*ft_real_dup3)(int, int, int);
static int (*ft_real_fcntl)(int, int, ...);
static int (*ft_real_fcntl64)(int, int, ...);
static int (*ft_real_close_range)(unsigned int, unsigned int, int);
#define FT_RESOLVE(name) do { if (ft_real_##name == NULL) ft_real_##name = dlsym(RTLD_NEXT, #name); } while (0)

static inline bool ft_tfd_known(int fd)
{
  return fd >= 0 && fd < FT_TFD_CLOCKS && __atomic_load_n(&ft_tfd_clock[fd], __ATOMIC_RELAXED) != 0;
}
/* the process whose descriptors these tables describe: set at load and in the child of each glibc fork. A child that
   shares this memory without fork (vfork, as Python's subprocess uses, or a CLONE_VM clone) has a descriptor table of
   its own, so what it closes or duplicates before exec changes nothing here. A fork's child also takes the tables'
   lock free: another thread may have held it at the fork, and no thread of the child would ever release it. */
static volatile pid_t ft_fd_owner = 0;
static void ft_fd_forked(void)
{
  ft_fd_owner = getpid();
  ft_tab_lock_word = 0;
}
static inline bool ft_fd_mine(void)
{
  return getpid() == ft_fd_owner;
}
/* ``fd`` closed, or about to name another file. Lock held. */
static void ft_fd_gone_locked(int fd)
{
  struct ft_timer *t;
  ft_tfd_clock[fd] = 0;
  while ((t = ft_timer_find_fd(fd)) != NULL)
  {
    int k = 0;
    while (t->fds[k] != fd) k++;
    t->fds[k] = t->fds[--t->nfds];
    if (t->nfds == 0) t->kind = 0;
  }
}
static void ft_fd_gone(int fd)
{
  sigset_t old;
  if (!ft_tfd_known(fd) || !ft_fd_mine()) return;
  ft_tab_lock(&old);
  ft_fd_gone_locked(fd);
  ft_tab_unlock(&old);
}
/* ``to`` now names ``from``'s file, a known timerfd. Lock held. */
static void ft_fd_duped_locked(int from, int to)
{
  struct ft_timer *t;
  if (to < 0 || to >= FT_TFD_CLOCKS) return;
  ft_tfd_clock[to] = ft_tfd_clock[from];
  if ((t = ft_timer_find_fd(from)) != NULL)
  {
    if (t->nfds < FT_TIMER_FDS)
      t->fds[t->nfds++] = to;
    else
      t->kind = 0;
  }
}
static void ft_fd_duped(int from, int to)
{
  sigset_t old;
  if (!ft_tfd_known(from) || !ft_fd_mine()) return;
  ft_tab_lock(&old);
  ft_fd_duped_locked(from, to);
  ft_tab_unlock(&old);
}
/* dup2 or dup3 onto ``fd2``. One that replaces a timerfd (whose close doesn't block) makes the change and its
   bookkeeping under the lock, so a failed call keeps the record and the helper never arms the number in between. */
static int ft_dup_onto(int fd, int fd2, int flags, bool three)
{
  sigset_t old;
  int r;
  FT_RESOLVE(dup2);
  FT_RESOLVE(dup3);
  if (fd == fd2 || !ft_tfd_known(fd2) || !ft_fd_mine())
  {
    r = three ? (*ft_real_dup3)(fd, fd2, flags) : (*ft_real_dup2)(fd, fd2);
    if (r >= 0 && fd != fd2) ft_fd_duped(fd, r);
    return r;
  }
  ft_tab_lock(&old);
  r = three ? (*ft_real_dup3)(fd, fd2, flags) : (*ft_real_dup2)(fd, fd2);
  if (r >= 0)
  {
    ft_fd_gone_locked(fd2);
    if (ft_tfd_known(fd)) ft_fd_duped_locked(fd, fd2);
  }
  ft_tab_unlock(&old);
  return r;
}

int close(int fd)
{
  ft_fd_gone(fd);
  FT_RESOLVE(close);
  return (*ft_real_close)(fd);
}
int dup(int fd)
{
  int r;
  FT_RESOLVE(dup);
  r = (*ft_real_dup)(fd);
  if (r >= 0) ft_fd_duped(fd, r);
  return r;
}
int dup2(int fd, int fd2)
{
  return ft_dup_onto(fd, fd2, 0, false);
}
int dup3(int fd, int fd2, int flags)
{
  return ft_dup_onto(fd, fd2, flags, true);
}
/* the argument passed on as glibc's own fcntl reads it, a pointer-sized word whatever the command */
int fcntl(int fd, int cmd, ...)
{
  va_list ap;
  void *arg;
  int r;
  va_start(ap, cmd);
  arg = va_arg(ap, void *);
  va_end(ap);
  FT_RESOLVE(fcntl);
  r = (*ft_real_fcntl)(fd, cmd, arg);
  if (r >= 0 && (cmd == F_DUPFD || cmd == F_DUPFD_CLOEXEC)) ft_fd_duped(fd, r);
  return r;
}
int fcntl64(int fd, int cmd, ...)
{
  va_list ap;
  void *arg;
  int r;
  va_start(ap, cmd);
  arg = va_arg(ap, void *);
  va_end(ap);
  FT_RESOLVE(fcntl64);
  r = (*ft_real_fcntl64)(fd, cmd, arg);
  if (r >= 0 && (cmd == F_DUPFD || cmd == F_DUPFD_CLOEXEC)) ft_fd_duped(fd, r);
  return r;
}
/* one that closes a timerfd runs under the lock, as dup2 does, so a failed call keeps the records. In practice it runs
   in a child of fork before exec, where no other thread wants the lock. */
int close_range(unsigned int first, unsigned int last, int flags)
{
  sigset_t old;
  bool timerfd = false;
  int r;
  FT_RESOLVE(close_range);
  if (!(flags & CLOSE_RANGE_CLOEXEC) && ft_fd_mine())
    for (unsigned int fd = first; fd <= last && fd < FT_TFD_CLOCKS && !timerfd; fd++) timerfd = ft_tfd_known((int) fd);
  if (!timerfd) return (*ft_real_close_range)(first, last, flags);
  ft_tab_lock(&old);
  r = (*ft_real_close_range)(first, last, flags);
  if (r == 0)
    for (unsigned int fd = first; fd <= last && fd < FT_TFD_CLOCKS; fd++)
      if (ft_tfd_known((int) fd)) ft_fd_gone_locked((int) fd);
  ft_tab_unlock(&old);
  return r;
}
/* the owner, and the real calls resolved at load, so a first close() in a signal handler doesn't call dlsym */
static void __attribute__((constructor)) ft_fd_init(void)
{
  ft_fd_owner = getpid();
  pthread_atfork(NULL, NULL, ft_fd_forked);
  FT_RESOLVE(close);
  FT_RESOLVE(dup);
  FT_RESOLVE(dup2);
  FT_RESOLVE(dup3);
  FT_RESOLVE(fcntl);
  FT_RESOLVE(fcntl64);
  FT_RESOLVE(close_range);
}

/* ---- POSIX timers (unversioned: upstream's GLIBC_2.2 / 2.3.3 versions don't bind glibc 2.34's timer_settime) ---- */
int timer_create(clockid_t clock, struct sigevent *sev, timer_t *id)
{
  int r;
  ftpl_init();
  if (real_timer_create_new == NULL) real_timer_create_new = dlsym(RTLD_NEXT, "timer_create");
  if (real_timer_create_new == NULL)
  {
    errno = ENOSYS;
    return -1;
  }
  r = (*real_timer_create_new)(clock, sev, id);
  if (r == 0 && id != NULL && ft_rt_clock(clock) && ft_faked())
  {
    sigset_t old;
    struct ft_timer *t;
    ft_tab_lock(&old);
    t = ft_timer_find_id(*id);
    if (t == NULL) t = ft_timer_new();
    if (t != NULL)
    {
      memset(t, 0, sizeof *t);
      t->kind = 2;
      t->id = *id;
      t->clock = clock;
    }
    ft_timers_any = 1;
    ft_tab_unlock(&old);
  }
  return r;
}
int timer_delete(timer_t id)
{
  ftpl_init();
  if (real_timer_delete_new == NULL) real_timer_delete_new = dlsym(RTLD_NEXT, "timer_delete");
  if (ft_timers_any)
  {
    sigset_t old;
    struct ft_timer *t;
    ft_tab_lock(&old);
    if ((t = ft_timer_find_id(id)) != NULL) t->kind = 0;
    ft_tab_unlock(&old);
  }
  return (*real_timer_delete_new)(id);
}
int timer_settime(timer_t id, int flags, const struct itimerspec *nv, struct itimerspec *ov)
{
  struct ft_clock_page s;
  struct ft_timer *t;
  struct itimerspec prev;
  sigset_t old;
  bool was = false;
  int r;
  ftpl_init();
  if (real_timer_settime_new == NULL) real_timer_settime_new = dlsym(RTLD_NEXT, "timer_settime");
  if (real_timer_gettime_new == NULL) real_timer_gettime_new = dlsym(RTLD_NEXT, "timer_gettime");
  if (!ft_timers_any || nv == NULL) return (*real_timer_settime_new)(id, flags, nv, ov);
  ft_tab_lock(&old);
  t = ft_timer_find_id(id);
  if (t != NULL && t->tracked)
  {
    ft_timer_value(t, &prev);
    was = true;
    t->tracked = 0;
  }
  ft_tab_unlock(&old);
  if (t == NULL || !ft_uw_active() || !(flags & TIMER_ABSTIME) || !ft_valid_ts(&nv->it_value)
      || !ft_valid_ts(&nv->it_interval) || (nv->it_value.tv_sec == 0 && nv->it_value.tv_nsec == 0) || !ft_snap(&s))
  {
    r = (*real_timer_settime_new)(id, flags, nv, ov);
    if (r == 0 && ov != NULL && was) *ov = prev;
    return r;
  }
  ft_tab_lock(&old);
  if (ov != NULL)
  {
    if (was)
      *ov = prev;
    else
      (*real_timer_gettime_new)(id, ov);
  }
  t->due = ft_ns(&nv->it_value);
  t->interval = ft_ns(&nv->it_interval);
  t->tracked = 1;
  r = ft_timer_arm(t, &s, true);
  ft_tab_unlock(&old);
  if (r == 0) ft_helper_start();
  return r;
}
int timer_gettime(timer_t id, struct itimerspec *cur)
{
  sigset_t old;
  struct ft_timer *t;
  ftpl_init();
  if (real_timer_gettime_new == NULL) real_timer_gettime_new = dlsym(RTLD_NEXT, "timer_gettime");
  if (ft_timers_any && cur != NULL)
  {
    ft_tab_lock(&old);
    t = ft_timer_find_id(id);
    if (t != NULL && t->tracked)
    {
      ft_timer_value(t, cur);
      ft_tab_unlock(&old);
      return 0;
    }
    ft_tab_unlock(&old);
  }
  return (*real_timer_gettime_new)(id, cur);
}

