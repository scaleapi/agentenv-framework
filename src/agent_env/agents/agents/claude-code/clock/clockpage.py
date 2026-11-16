"""The follower's side of the clock page (version 2) and of the clock file: writes the universe's realtime
clock to both. Elapsed time is real, so the page's monotonic fields are written as the real monotonic clock (identity)
and no library reads them; they stay so the page keeps its 80-byte layout.

  clockpage.py set UNIVERSE RATE [ANCHOR]   armed: UNIVERSE (epoch s) at real time ANCHOR (default now), at RATE
  clockpage.py clear                        real time
  clockpage.py create OWNER                 as root at boot: the page, OWNER's from its first byte, unarmed
  clockpage.py show

The page (default /dev/shm/agentenv-clock, a tmpfs, so readers that mapped it see every write) is created once, 0644,
owned by whoever writes it, and rewritten in place: seq odd, the fields and their check, seq even, then FUTEX_WAKE on
seq, which wakes every wait agent-env's libfaketime has for a clock time, and its helper threads, so they plan again.
Readers never see a torn value: they retry (waiting on seq) while seq is odd or changed, or the check doesn't match.
A page a writer died writing (seq left odd) is taken up by the next write, which then writes the universe as given.

A write that would put the universe less than CONTINUITY_NS behind where the page has it now (a rate change computed
from a universe read a moment before the write) continues it from where it is instead: the write's real time is taken
last, while seq is odd, and the universe at it read from the page being replaced, so no reader of either page ever sees
the clock step back. A larger step back is a set_time and is written as given."""
import ctypes
import mmap
import os
import platform
import pwd
import stat
import struct
import sys
import time
from decimal import Decimal

PAGE = os.environ.get("AGENTENV_CLOCK_PAGE", "/dev/shm/agentenv-clock")
FILE = os.environ.get("AGENTENV_CLOCK_FILE", "/run/agentenv-clock/clock")
MAGIC, VERSION = 0x4B434541, 2
# magic version seq universe_ns anchor_ns rate armed reserved generation check mono_universe_ns mono_anchor_ns
LAYOUT = struct.Struct("<IIQqqdIIQQqq")
MASK = 0xFFFFFFFFFFFFFFFF
CONTINUITY_NS = 1_000_000_000
_SYS_FUTEX = ({"x86_64": 202, "aarch64": 98, "arm64": 98}.get(platform.machine())
              if platform.system() == "Linux" else None)
_FUTEX_WAKE = 1
_libc = ctypes.CDLL(None, use_errno=True)


def check(seq, universe_ns, anchor_ns, rate, armed, generation, mono_universe_ns, mono_anchor_ns) -> int:
    rate_bits = struct.unpack("<Q", struct.pack("<d", rate))[0]
    return (seq ^ (universe_ns & MASK) ^ (anchor_ns & MASK) ^ rate_bits ^ armed ^ generation
            ^ (mono_universe_ns & MASK) ^ (mono_anchor_ns & MASK) ^ 0x9E3779B97F4A7C15) & MASK


def encode(seq, universe_ns, anchor_ns, rate, armed, generation, mono_universe_ns, mono_anchor_ns) -> bytes:
    """The page's bytes for these fields, its check among them: what every write writes and every reader checks,
    agent-env's libfaketime and the egress proxy (its egress/testdata pins them)."""
    return LAYOUT.pack(MAGIC, VERSION, seq, universe_ns, anchor_ns, rate, armed, 0, generation,
                       check(seq, universe_ns, anchor_ns, rate, armed, generation, mono_universe_ns, mono_anchor_ns),
                       mono_universe_ns, mono_anchor_ns)


def open_page() -> mmap.mmap:
    try:
        fd = os.open(PAGE, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        os.ftruncate(fd, LAYOUT.size)
        os.write(fd, encode(0, 0, 0, 1.0, 0, 0, 0, 0))
    except FileExistsError:
        fd = os.open(PAGE, os.O_RDWR)
        if os.fstat(fd).st_uid != os.getuid():
            raise SystemExit(f"{PAGE} isn't ours")
        if os.fstat(fd).st_size < LAYOUT.size:
            os.close(fd)
            raise SystemExit(f"{PAGE} is an older page")
    m = mmap.mmap(fd, LAYOUT.size)
    os.close(fd)
    return m


def _wake(m: mmap.mmap) -> None:
    """FUTEX_WAKE every waiter on seq's low word (a shared futex: other processes' read-only mappings are woken too)."""
    if _SYS_FUTEX is None:
        return
    buf = (ctypes.c_char * LAYOUT.size).from_buffer(m)
    try:
        _libc.syscall(ctypes.c_long(_SYS_FUTEX), ctypes.c_void_p(ctypes.addressof(buf) + 8), ctypes.c_int(_FUTEX_WAKE),
                      ctypes.c_int(0x7FFFFFFF), None, None, ctypes.c_int(0))
    finally:
        del buf


def _universe_at(armed, universe_ns, anchor_ns, rate, now_ns) -> int:
    if not armed:
        return now_ns
    if rate == 0 or now_ns <= anchor_ns:
        return universe_ns
    return universe_ns + int((now_ns - anchor_ns) * rate)


def write(armed: int, universe: float, rate: float, anchor: float) -> dict:
    m = open_page()
    _, _, seq, o_u, o_a, o_rate, o_armed, _, generation, _, _, _ = LAYOUT.unpack(m[:LAYOUT.size])
    universe_ns, anchor_ns = int(Decimal(str(universe)) * 10**9), int(Decimal(str(anchor)) * 10**9)
    if not armed:
        rate = 1.0
    generation += 1
    dead = seq & 1  # a writer died mid-write: its fields may be torn, so this write goes on from the next even seq
    seq += dead
    m[8:16] = struct.pack("<Q", seq + 1)
    # The real time last, while seq is odd: every reader of the old page read its real time before it.
    now_rt = time.time_ns()
    now_mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    old_now = _universe_at(o_armed, o_u, o_a, o_rate, now_rt)
    new_now = _universe_at(armed, universe_ns, anchor_ns, rate, now_rt)
    continued = armed and generation > 1 and not dead and 0 < old_now - new_now < CONTINUITY_NS
    if continued:
        universe_ns, anchor_ns = old_now, now_rt
    body = encode(seq + 2, universe_ns, anchor_ns, rate, armed, generation, now_mono, now_mono)
    m[16:] = body[16:]
    m[8:16] = struct.pack("<Q", seq + 2)
    _wake(m)
    m.close()
    spec = f"@{universe_ns / 1e9:.9f} x{rate:g} ^{anchor_ns / 1e9:.9f}" if armed else "+0 x1"
    os.makedirs(os.path.dirname(FILE), exist_ok=True)
    with open(FILE + ".part", "w") as f:
        f.write(spec + "\n")
    os.replace(FILE + ".part", FILE)
    return {"generation": generation, "spec": spec, "jump_ns": new_now - old_now if generation > 1 else 0,
            "continued": bool(continued), "universe_ns": universe_ns, "anchor_ns": anchor_ns}


def create(owner: str) -> None:
    """As root at boot: the page, ``owner``'s from its first byte (unarmed), replacing one anyone else made."""
    user = pwd.getpwnam(owner)
    try:
        st = os.lstat(PAGE)
        if stat.S_ISREG(st.st_mode) and st.st_uid == user.pw_uid and st.st_size == LAYOUT.size:
            return
        os.unlink(PAGE)
    except FileNotFoundError:
        pass
    fd = os.open(PAGE, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.fchown(fd, user.pw_uid, user.pw_gid)
        os.write(fd, encode(0, 0, 0, 1.0, 0, 0, 0, 0))
        os.fchmod(fd, 0o644)
    finally:
        os.close(fd)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "set":
        anchor = float(sys.argv[4]) if len(sys.argv) > 4 else time.time()
        print(write(1, float(sys.argv[2]), float(sys.argv[3]), anchor))
    elif cmd == "create":
        create(sys.argv[2])
    elif cmd == "clear":
        print(write(0, 0.0, 1.0, time.time()))
    else:
        with open(PAGE, "rb") as f:
            print(LAYOUT.unpack(f.read(LAYOUT.size)))
