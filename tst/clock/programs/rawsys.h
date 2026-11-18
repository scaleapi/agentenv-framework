/* rawsys.h: system calls through the raw instruction (x86_64 or aarch64), which libfaketime can't see. */
#include <time.h>
static inline long raw_syscall2(long n_x86, long n_arm, long a, long b)
{
#if defined(__aarch64__)
  register long x8 __asm__("x8") = n_arm;
  register long x0 __asm__("x0") = a;
  register long x1 __asm__("x1") = b;
  __asm__ volatile("svc 0" : "+r"(x0) : "r"(x8), "r"(x1) : "memory");
  return x0;
#else
  long r;
  __asm__ volatile("syscall" : "=a"(r) : "0"(n_x86), "D"(a), "S"(b) : "rcx", "r11", "memory");
  return r;
#endif
}
/* A relative sleep the library can't stretch or shorten. */
static inline long raw_nanosleep(const struct timespec *t) { return raw_syscall2(35, 101, (long) t, 0); }
/* The kernel's CLOCK_REALTIME, whatever the library makes of the libc call. */
static inline long raw_clock_gettime(clockid_t clock, struct timespec *t) { return raw_syscall2(228, 113, (long) clock, (long) t); }
