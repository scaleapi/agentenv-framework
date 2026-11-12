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
| `clock-follower` | the follower: keeps the page in step with the gateway's clock (below) | Apache-2.0 |
| `follow` | root's loop that keeps the follower running as `agentenv-clock` | Apache-2.0 |

Build it on x86_64 or aarch64 Linux with a C toolchain, make, patch and curl:

    src/agent_env/computers/clock/build.sh OUT [TARBALL]

## On a computer: the `clock` app and the follower

A computer env follows its multi env's gateway clock by composing the `clock` app (`../envs/clock`, the alias
`clock`): `apps = ["clock"]` on a Linux X11 computer, `amd64` or `arm64`. Its install builds this library at compose,
installs `clock.conf` and preloads the library into every process (`/etc/ld.so.preload`), and adds the follower, its
loop and a user of its own, `agentenv-clock`, whose uid `clock.conf` names as the page's owner, so the library trusts no
page another user made.

When `sync_env_clock` arms a multi env whose computer composes it, core configures the computer's follower before the
arm, as root, through the sandbox's root channel (`agent_env.env.envs.computer.utils.clock`), never through the driver,
which runs as the session user. It writes the follower's config (the gateway's `/clock/watch` URL as the computer's
sandbox reaches it, the computer's read token, its follower name and a config id) to
`/etc/agentenv/clock-follower/follower.json`, the follower's alone to read; makes the page, the follower's from its first
byte; and starts `follow`, which runs `clock-follower` as `agentenv-clock` and runs it again within a second whenever it
exits. The arm then waits for the computer to show the clock (`wait_followers_s`) and fails the step naming it if it
doesn't; the app's `clock/v1` `sync_time` reports what the follower showed for that config id.

Each round, the follower long-polls `/clock/watch` with the version it last showed, which acknowledges that version. On a
new version it writes the page: the universe the gateway reported, at its rate, from the local real time the gateway's
`sent` stamp corresponds to, the offset between the two clocks estimated from the watch's four stamps as NTP does. It
records what it showed in `/run/agentenv-clock/status.json`. A failed round is recorded there and retried with backoff,
and a new config starts it over from no version.

The session user can't write the page or the clock file, read the follower's config, or stop either process: each is
another user's. A multi env's reset (`reset_env`) boots a computer afresh, follower and all; while the clock is armed,
core starts the new follower as an arm does and the reset ends once the computer shows the clock.

Not covered yet: agent-env's own servers run on universe time too, with no real-time marker; and certificates checked
by TLS stacks the shim doesn't reach (Chromium's), Chromium's time zone and an egress proxy for live sites at other dates
are later work.

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
live sites verify, two local servers show the check is on real time and expired and untrusted certificates are still
refused, and two copies of OpenSSL in one process, the system's 3.0 and a 1.1, each check with their own), and
Chromium launches. `.github/workflows/clock.yml` runs them on x86_64 and arm64. The gates hold correctness strictly
(no wait ends early, every wait returns what it should, nothing is lost or torn) and lateness to
`tst/clock/bounds.py`, a bound a broken library misses by seconds and a slow machine doesn't reach.
`tst/integration/computer/clock_desktop_test.py` composes the library into a real computer (the ubuntu-24.04 desktop,
the reference driver and agentenv's chromium, every process preloading it), which must pass the conformance probe at
full; then, its page written as root, a shell, Chrome's JavaScript and the driver's input follow the universe through
30 jumps, a rate change and a freeze, with Chrome, Xvfb and the desktop's panel running on.
