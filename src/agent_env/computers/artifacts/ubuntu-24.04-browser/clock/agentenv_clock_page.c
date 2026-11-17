/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * agent-env's additions to libfaketime 0.9.12, part 1, included by libfaketime.c (libfaketime.patch) before its
 * time functions: the process's settings, read once at load from /etc/agentenv/clock.conf, or from the environment
 * where there is no such file, so no getenv() runs after main() starts; who is faked (the session user, unless started with AGENTENV_REAL_TIME=1, a marker
 * taken out of the environment block in place); and the clock page the follower writes, read without a cache.
 * Derived from libfaketime, so licensed as it is, under the GNU General Public License version 2 (COPYING).
 */
/*
 * configuration is read once, when the library loads, from AGENTENV_CLOCK_CONF (NAME=VALUE lines) when that
 * file exists, else from the environment, and never with getenv() afterwards: a getenv() racing a program's setenv()
 * reads a freed environ array, which is how libfaketime's pthread_cond_timedwait crashed Chromium.
 */
#ifndef AGENTENV_CLOCK_CONF
#define AGENTENV_CLOCK_CONF "/etc/agentenv/clock.conf"
#endif
#define FT_CONF_MAX 64
static char ft_conf_text[8192];
static const char *ft_conf_name[FT_CONF_MAX];
static const char *ft_conf_value[FT_CONF_MAX];
static int ft_conf_count = 0;
static bool ft_conf_loaded = false;
/* this process stays on real time: another user's than AGENTENV_SESSION_UID, or started with
   AGENTENV_REAL_TIME=1. */
static bool ft_disabled = false;
/* bumped with each new time specification parsed, so a wait sees the clock change under it. */
static volatile unsigned long ft_spec_gen = 0;
static __thread unsigned long ft_tl_gen = 0;
/* real seconds per fake second for what is left of a wait, as of this thread's last clock read: 1 / rate,
   or 1 for a frozen clock. */
static __thread double ft_tl_scale = 1.0;
/* how long a wait that can be cut sleeps before it looks at the clock again (AGENTENV_WAIT_SLICE_MS). */
static long ft_wait_slice_ns = 100000000L;

static void ft_conf_add(char *name, char *value)
{
  if (ft_conf_count < FT_CONF_MAX)
  {
    ft_conf_name[ft_conf_count] = name;
    ft_conf_value[ft_conf_count] = value;
    ft_conf_count++;
  }
}

static void ft_conf_load(void)
{
  extern char **environ;
  size_t used = 0;
  int fd;
  if (ft_conf_loaded) return;
  ft_conf_loaded = true;
  fd = open(AGENTENV_CLOCK_CONF, O_RDONLY | O_CLOEXEC);
  if (fd >= 0)
  {
    ssize_t n;
    char *line;
    while (used < sizeof(ft_conf_text) - 1 &&
           (n = read(fd, ft_conf_text + used, sizeof(ft_conf_text) - 1 - used)) > 0)
      used += n;
    close(fd);
    ft_conf_text[used] = 0;
    line = ft_conf_text;
    while (line != NULL && *line != 0)
    {
      char *next = strchr(line, '\n');
      char *eq;
      if (next != NULL) *next++ = 0;
      eq = strchr(line, '=');
      if (line[0] != '#' && eq != NULL && eq != line)
      {
        *eq = 0;
        ft_conf_add(line, eq + 1);
      }
      line = next;
    }
    return;
  }
  for (char **e = environ; e != NULL && *e != NULL; e++)
  {
    const char *s = *e;
    size_t len;
    char *copy, *eq;
    if (strncmp(s, "FAKE", 4) && strncmp(s, "NO_FAKE", 7) && strncmp(s, "DONT_FAKE", 9) && strncmp(s, "HOME=", 5)
        && strncmp(s, "AGENTENV_", 9))
      continue;
    len = strlen(s);
    if (used + len + 1 > sizeof(ft_conf_text)) break;
    copy = ft_conf_text + used;
    memcpy(copy, s, len + 1);
    used += len + 1;
    eq = strchr(copy, '=');
    if (eq == NULL) continue;
    *eq = 0;
    ft_conf_add(copy, eq + 1);
  }
}

static char *ft_getenv(const char *name)
{
  ft_conf_load();
  for (int i = 0; i < ft_conf_count; i++)
  {
    if (strcmp(ft_conf_name[i], name) == 0) return (char *) ft_conf_value[i];
  }
  return NULL;
}

/*
 * AGENTENV_REAL_TIME=1 taken out of the original environment block in place: its pointer out of the array
 * (which is main()'s envp too) and the string itself changed to "=0", so no copy of it is the marker. unsetenv() isn't
 * used: a program may define its own (bash does, and before its main() it does nothing).
 */
static void ft_scrub_marker(void)
{
  extern char **environ;
  char **e = environ;
  if (e == NULL) return;
  while (*e != NULL)
  {
    if (strcmp(*e, "AGENTENV_REAL_TIME=1") == 0)
    {
      char **d = e;
      (*e)[sizeof("AGENTENV_REAL_TIME=") - 1] = '0';
      do d[0] = d[1]; while (*d++ != NULL);
    }
    else
      e++;
  }
}

/* whether a wait's deadline is one to translate: a valid one, in a faked process, outside libfaketime. */
static bool ft_translates_wait(const struct timespec *abstime)
{
  return abstime != NULL && !ft_disabled && !dont_fake && abstime->tv_nsec >= 0 && abstime->tv_nsec < SEC_TO_nSEC;
}

/*
 * where a CLOCK_REALTIME deadline in fake time falls on a real clock (CLOCK_MONOTONIC or CLOCK_REALTIME):
 * now, plus what is left of the fake interval times ft_tl_scale; so a frozen clock counts what is left as real time and
 * nothing waits forever, and a deadline already past is now.
 */
static void ft_deadline_from_fake(clockid_t clock, const struct timespec *fake_abs, struct timespec *out)
{
  struct timespec now, real_now, fake_now, left, scaled;
  DONT_FAKE_TIME((*real_clock_gettime)(clock, &now));
  DONT_FAKE_TIME((*real_clock_gettime)(CLOCK_REALTIME, &real_now));
  fake_now = real_now;
  (void) fake_clock_gettime(CLOCK_REALTIME, &fake_now);
  *out = now;
  timespecsub(fake_abs, &fake_now, &left);
  if (left.tv_sec < 0) return;
  if (left.tv_sec > 100L * 365 * 86400) left.tv_sec = 100L * 365 * 86400;
  if (ft_tl_scale != 1.0)
  {
    timespecmul(&left, ft_tl_scale, &scaled);
  }
  else
  {
    scaled = left;
  }
  timespecadd(&now, &scaled, out);
}

/* the end of a wait's next slice on CLOCK_MONOTONIC; true when that is the deadline itself. */
static bool ft_slice_end(const struct timespec *mono_abs, struct timespec *slice)
{
  struct timespec now, step = {ft_wait_slice_ns / SEC_TO_nSEC, ft_wait_slice_ns % SEC_TO_nSEC};
  DONT_FAKE_TIME((*real_clock_gettime)(CLOCK_MONOTONIC, &now));
  timespecadd(&now, &step, slice);
  if (timespeccmp(slice, mono_abs, <)) return false;
  *slice = *mono_abs;
  return true;
}

/* after a slice, the clock read again (the file, with no cache); a new specification since ``gen`` moves the
   deadline. True when it moved. */
static bool ft_deadline_moved(const struct timespec *fake_abs, struct timespec *mono_abs, unsigned long *gen)
{
  struct timespec now;
  DONT_FAKE_TIME((*real_clock_gettime)(CLOCK_REALTIME, &now));
  (void) fake_clock_gettime(CLOCK_REALTIME, &now);
  if (ft_tl_gen == *gen) return false;
  *gen = ft_tl_gen;
  ft_deadline_from_fake(CLOCK_MONOTONIC, fake_abs, mono_abs);
  return true;
}

/*
 * the clock page. 64 bytes, little-endian, rewritten in place by the follower: seq odd while it writes,
 * then even; check is the xor of the other words and a constant. A realtime read in a faked process copies it
 * lock-free and computes universe + (now - anchor) * rate (rate 0: universe), so every process agrees with no shared
 * start; the first thread to see a new generation also hands it to libfaketime's own state (rates of sleeps, the
 * time an absolute spec gives) under the MT lock. The page must be owned by AGENTENV_CLOCK_PAGE_UID (0 by default) and
 * writable by no one else, so the session user can't plant one.
 */
struct ft_clock_page
{
  uint32_t magic, version;
  uint64_t seq;
  int64_t universe_ns, anchor_ns;
  double rate;
  uint32_t armed, reserved;
  uint64_t generation, check;
  int64_t mono_universe_ns, mono_anchor_ns;
};
#define FT_PAGE_MAGIC 0x4B434541u
#define FT_PAGE_VERSION 2u
static const volatile struct ft_clock_page *ft_page = NULL;
static uint64_t ft_page_applied = 0;
static time_t ft_page_last_try = 0;
static void parse_ft_string(const char *user_faked_time);

static uint64_t ft_page_check(const struct ft_clock_page *p)
{
  uint64_t rate_bits;
  memcpy(&rate_bits, &p->rate, sizeof rate_bits);
  return p->seq ^ (uint64_t) p->universe_ns ^ (uint64_t) p->anchor_ns ^ rate_bits ^ p->armed ^ p->generation
         ^ (uint64_t) p->mono_universe_ns ^ (uint64_t) p->mono_anchor_ns ^ 0x9E3779B97F4A7C15ull;
}

static bool ft_page_snapshot(struct ft_clock_page *out)
{
  for (int i = 0; i < 1000; i++)
  {
    uint64_t s1 = __atomic_load_n(&ft_page->seq, __ATOMIC_ACQUIRE);
    if (s1 & 1) continue;
    out->magic = ft_page->magic;
    out->seq = s1;
    out->universe_ns = ft_page->universe_ns;
    out->anchor_ns = ft_page->anchor_ns;
    out->rate = ft_page->rate;
    out->armed = ft_page->armed;
    out->version = ft_page->version;
    out->generation = ft_page->generation;
    out->check = ft_page->check;
    out->mono_universe_ns = ft_page->mono_universe_ns;
    out->mono_anchor_ns = ft_page->mono_anchor_ns;
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if (__atomic_load_n(&ft_page->seq, __ATOMIC_RELAXED) == s1 && out->magic == FT_PAGE_MAGIC
        && out->version == FT_PAGE_VERSION
        && ft_page_check(out) == out->check)
      return true;
  }
  return false;
}

static void ft_page_open(void)
{
  const char *path = ft_getenv("AGENTENV_CLOCK_PAGE");
  const char *owner = ft_getenv("AGENTENV_CLOCK_PAGE_UID");
  struct stat st;
  void *map;
  int fd;
  if (path == NULL || *path == 0) return;
  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return;
  if (fstat(fd, &st) != 0 || st.st_size < (off_t) sizeof(struct ft_clock_page)
      || st.st_uid != (uid_t) (owner ? strtoul(owner, NULL, 10) : 0) || (st.st_mode & 022) != 0)
  {
    close(fd);
    return;
  }
  map = mmap(NULL, sizeof(struct ft_clock_page), PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (map == MAP_FAILED) return;
  if (((const struct ft_clock_page *) map)->magic != FT_PAGE_MAGIC
      || ((const struct ft_clock_page *) map)->version != FT_PAGE_VERSION)
  {
    munmap(map, sizeof(struct ft_clock_page));
    return;
  }
  ft_page = map;
}

/* libfaketime's own state set to the page's clock; called with the MT lock held. */
static void ft_page_apply_locked(const struct ft_clock_page *s)
{
  char spec[160];
  if (s->generation == ft_page_applied) return;
  if (!s->armed)
    snprintf(spec, sizeof spec, "+0 x1");
  else
    snprintf(spec, sizeof spec, "@%lld.%09lld x%.17g ^%lld.%09lld",
             (long long) (s->universe_ns / 1000000000LL), (long long) (s->universe_ns % 1000000000LL), s->rate,
             (long long) (s->anchor_ns / 1000000000LL), (long long) (s->anchor_ns % 1000000000LL));
  parse_ft_string(spec);
  __atomic_store_n(&ft_page_applied, s->generation, __ATOMIC_RELEASE);
}

/* "%s" times are read as a number, without strptime() and mktime(), which reads TZ with getenv(). */
static const char *ft_parse_epoch(const char *s, struct timespec *out)
{
  char *end;
  long long sec;
  long nsec = 0, scale = 100000000L;
  bool negative;
  while (*s == ' ') s++;
  negative = (*s == '-');
  sec = strtoll(s, &end, 10);
  if (end == s) return NULL;
  if (*end == '.')
  {
    end++;
    while (*end >= '0' && *end <= '9')
    {
      nsec += (*end - '0') * scale;
      scale /= 10;
      end++;
    }
  }
  if (negative && nsec > 0)
  {
    sec -= 1;
    nsec = SEC_TO_nSEC - nsec;
  }
  out->tv_sec = (time_t) sec;
  out->tv_nsec = nsec;
  return end;
}

