/* rtbase.h: the universe's realtime as 64-bit nanoseconds counted from the second it was first read, so a deadline
   computed near either end of the clock's range stays inside 64 bits, and rt_ts turns one back into a valid time before
   1970 too (its nanoseconds floored, never negative). Read it once before starting threads. */
#include <stdint.h>
#include <time.h>
static int64_t rt_base_s;
static int rt_based;
static inline int64_t rt_ns(void)
{
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  if (!rt_based) { rt_base_s = t.tv_sec; rt_based = 1; }
  return (int64_t) (t.tv_sec - rt_base_s) * 1000000000LL + t.tv_nsec;
}
static inline struct timespec rt_ts(int64_t v)
{
  int64_t s = v / 1000000000LL, n = v % 1000000000LL;
  struct timespec t;
  if (n < 0) { s -= 1; n += 1000000000LL; }
  t.tv_sec = (time_t) (rt_base_s + s);
  t.tv_nsec = (long) n;
  return t;
}
