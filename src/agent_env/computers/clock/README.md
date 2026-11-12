# The computer's clock: agent-env's libfaketime

Every program on a computer reads its own clock, so a computer can't ask the gateway for universe time the way an MCP
server does. Instead each session program loads agent-env's build of [libfaketime](https://github.com/wolfcw/libfaketime)
(through `/etc/ld.so.preload`), which answers clock reads with universe time from a page a follower process keeps in
step with the gateway.

What follows the universe, and what doesn't:

- **Universe time:** clock reads (`CLOCK_REALTIME`, `_COARSE`, `_ALARM`, `CLOCK_TAI`, `gettimeofday`, `time`) and waits
  for a set realtime clock time (`clock_nanosleep` with `TIMER_ABSTIME`, realtime futex waits, `sem_timedwait` and
  `sem_clockwait`, realtime condition variables, absolute realtime timerfds and POSIX timers). A wait is planned in real
  time and planned again whenever the page changes, so a jump past its deadline ends it at the jump. Where the kernel
  has no `futex_waitv` (before 5.16, or gVisor), realtime futex and semaphore waits read the page between slices of
  `AGENTENV_WAIT_SLICE_MS` (100 ms by default) instead, so they see a change within a slice.
- **Real time:** the monotonic and boot clocks, CPU-time clocks, and every relative sleep and timeout, at any rate.
- **Certificates:** OpenSSL checks a peer's certificates against real time, through the TLS shim (`tls_realtime.c`,
  preloaded after the library), so a client that reaches a live site without the computer's egress proxy still
  verifies it months from real time, and still refuses an expired or untrusted one. The shim sets
  `X509_verify_cert`'s check time unless the caller set one, and calls the caller's own libcrypto. TLS stacks of
  their own (Chromium's BoringSSL, GnuTLS, NSS, Java's) and static binaries aren't covered.
- **Who:** only the session user's processes (`AGENTENV_SESSION_UID`). A program started with `AGENTENV_REAL_TIME=1`
  stays real; the marker is taken out of its environment block, so its children are faked.
- **Settings:** read once as the library loads, from `/etc/agentenv/clock.conf` (`clock.conf` is the default), or from
  the environment where there is no such file. While the file exists the environment changes none of them; the marker
  above is the one switch a program's environment has. No `getenv()` runs once the program has started: a `getenv()`
  racing a program's own `setenv()` made stock libfaketime crash Chromium on about one launch in five.

## Files

| File | What it is | License |
|---|---|---|
| `libfaketime.patch` | agent-env's changes to upstream 0.9.12's `src/` (applied with `patch -p1`) | GPL-2.0-only |
| `agentenv_clock_page.c` | settings, who is faked, the clock page reader; included by the patched `libfaketime.c` | GPL-2.0-only |
| `agentenv_clock_waits.c` | universe-time reads and waits, the per-process helper thread; included the same way | GPL-2.0-only |
| `COPYING` | the GNU General Public License version 2, libfaketime's license | |
| `build.sh` | builds `libfaketimeMT.so.1` from upstream's sha256-pinned tarball and the files above, and the shim's `libtls_realtime.so` | Apache-2.0 |
| `tls_realtime.c` | the TLS shim: OpenSSL's certificate checks on real time, preloaded after the library | Apache-2.0 |
| `clockpage.py` | the clock page's writer, which the follower uses | Apache-2.0 |
| `clock.conf` | the library's default settings | Apache-2.0 |

Build it on x86_64 or aarch64 Linux with a C toolchain, make, patch and curl:

    src/agent_env/computers/clock/build.sh OUT [TARBALL]

## The clock page

An 80-byte page (default `/dev/shm/agentenv-clock`, version 2) that `clockpage.py` rewrites in place: `seq` odd, the
fields and their check, `seq` even, then `FUTEX_WAKE` on `seq`, which wakes every wait the library has planned and its
helper threads. Readers never cache it and never see a torn value: they wait on `seq` while it is odd and retry when it
changed or the check doesn't match. A write that would step the universe back by less than a second (a rate change
computed from a read a moment earlier) continues from where the page has it instead.

## Tests

`tst/clock` (`make clock-test`) builds this library and stock 0.9.12 in an Ubuntu 24.04 container and runs its gates
there: every wait API in C, Python, Java and Node at several rates, jumps, cancelled waits, timerfd duplicates,
condition-variable and semaphore stress, the waits again in a container that refuses `futex_waitv` (as kernels before
5.16 and gVisor do), a stalled writer, one that died mid-write, torn reads, the marker, the environment race, forks
while another thread closes a timerfd, the TLS shim (curl, Python and `openssl s_client` with the clock 182 days back:
live sites verify, badssl's expired and untrusted ones don't, and two local servers show the check is on real time),
and Chromium launches. `.github/workflows/clock.yml` runs them on x86_64 and arm64. The gates hold correctness strictly (no wait ends early, every wait returns what it should, nothing is lost or
torn) and lateness to `tst/clock/bounds.py`, a bound a broken library misses by seconds and a slow machine doesn't
reach. `tst/integration/computer/clock_desktop_test.py` composes the library into a real computer (the ubuntu-24.04
desktop, the reference driver and agentenv's chromium, every process preloading it), which must pass the conformance
probe at full; then, its page written as root, a shell, Chrome's JavaScript and the driver's input follow the universe
through 30 jumps, a rate change and a freeze, with Chrome, Xvfb and the desktop's panel running on.
