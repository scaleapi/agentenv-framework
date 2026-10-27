"""snapshot/v1's changelog: each tool call's writes captured, and replayed onto a fresh agent.

``enable`` writes a capture script and baselines a GNU incremental tar of the capture roots; the CLI's post-tool hook
then archives each tool call's writes and uploads them through agent-env's namespace grant as
``<root_path>/<position:06d>.tgz``. ``apply`` extracts granted increments in order onto a fresh agent of the same
image, and can make the restored conversation resumable.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path

from agentenv_protocol.a2a_agent import (
    ChangelogIncrement,
    NamespaceChangelogEnableRequest,
    NamespaceChangelogEnableResponse,
    ObjectChangelogApplyRequest,
    ObjectChangelogApplyResponse,
)
from agentenv_protocol.transfers import download
from starlette.exceptions import HTTPException

logger = logging.getLogger(__name__)

# A capture hook can wait its turn behind other hooks (parallel tool calls), then tar and upload with retries; a CLI
# that kills it past its timeout leaves that tool call's increment missing.
HOOK_TIMEOUT_S = 600
_DIR_PREFIX = "gemini-changelog-"
META_NAME = "gemini-changelog-meta.json"
META_DIR_SUFFIX = ".changelog-meta"
# A capture's own metadata in an increment: META_NAME in a directory named with META_DIR_SUFFIX, wherever the
# capturing agent kept it, so a user file of that name elsewhere is restored like any other.
_META_MEMBER = re.compile(rf"(?:^|/)[^/]+{re.escape(META_DIR_SUFFIX)}/{re.escape(META_NAME)}\Z")
_TAR_STDERR_TAIL_CHARS = 2000
_TAR_VERSION_TIMEOUT_S = 30
_FAILED_INCREMENTS_LOGGED = 5

# The post-tool hook, run by the agent's own Python, as a child of the CLI after every tool call, so an increment
# includes that tool's writes. It exits 0 on a capture error, so the error never reaches the model.
CAPTURE_SCRIPT = r'''import asyncio, fcntl, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

from agentenv_protocol.transfers import NamespaceUploader, WriteNamespaceGrant

CFG = "/tmp/changelog/config.json"
STDERR_TAIL_CHARS = 500
TURN_POLL_S = 0.05
TOOL_NAME_CHARS = 40
# tar's exit status when a file changed as it read it: the archive is still whole.
TAR_FILE_CHANGED_RC = 1
SIGTERM_EXIT_CODE = 143


def _read_int(path):
    try:
        return int(open(path).read().strip())
    except Exception:
        return 0


def _write_atomic(path, text):
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _epoch():
    try:
        return json.load(open(CFG)).get("epoch")
    except Exception:
        return None


def _alive(changelog_dir, k):
    """Whether the hook holding position k is still running (the CLI kills one past its timeout)."""
    try:
        pid = int(open(os.path.join(changelog_dir, "held.%d" % k)).read())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _committed(changelog_dir):
    try:
        return json.load(open(os.path.join(changelog_dir, "namespace.json")))
    except Exception:
        return {}


def _reserve(changelog_dir, grant, name, size):
    """Reserve an object's bytes against the grant's limits, which every hook process shares."""
    with open(os.path.join(changelog_dir, "namespace.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prospective = {**_committed(changelog_dir), name: size}
        if (size > grant.max_object_bytes or len(prospective) > grant.max_objects
                or sum(prospective.values()) > grant.max_total_bytes):
            return False
        _write_atomic(os.path.join(changelog_dir, "namespace.json"), json.dumps(prospective))
        return True


def _release(changelog_dir, name):
    with open(os.path.join(changelog_dir, "namespace.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        committed = _committed(changelog_dir)
        committed.pop(name, None)
        _write_atomic(os.path.join(changelog_dir, "namespace.json"), json.dumps(committed))


def _store(changelog_dir, cfg, k, local):
    """Upload position k's archive through agent-env's namespace grant; False when the grant's limits refuse it."""
    grant = WriteNamespaceGrant.model_validate(cfg["write_namespace"])
    name = "%06d.tgz" % k
    if not _reserve(changelog_dir, grant, name, os.path.getsize(local)):
        sys.stderr.write("[changelog] skipped seq=%d: past the grant's limits\n" % k)
        return False
    try:
        asyncio.run(NamespaceUploader(grant).upload(name, Path(local)))
    except BaseException:
        _release(changelog_dir, name)
        raise
    return True


def _read_json(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def _tail(stderr):
    return stderr.decode(errors="replace").strip()[-STDERR_TAIL_CHARS:]


def _write_meta(meta, k, record):
    """Write position k's metadata; False, logged, when it can't be, so the capture goes on without it."""
    try:
        _write_atomic(meta, json.dumps(record))
        return True
    except Exception as e:
        sys.stderr.write("[changelog] metadata write failed seq=%d; capturing without it: %s\n" % (k, e))
        return False


def main():
    cfg = json.load(open(CFG))
    changelog_dir, snar, roots, epoch = cfg["dir"], cfg["snar"], cfg["roots"], cfg.get("epoch")
    meta, present_file = cfg.get("meta"), os.path.join(changelog_dir, "present.json")
    try:
        hook = json.load(sys.stdin)
    except Exception:
        hook = {}
    tool = str(hook.get("tool_name") or "tool").replace("/", "_")[:TOOL_NAME_CHARS]
    session = hook.get("session_id") if isinstance(hook.get("session_id"), str) else None
    seqf = os.path.join(changelog_dir, "seq")
    turnf = os.path.join(changelog_dir, "turn")
    held = os.path.join(changelog_dir, "held.%d")
    # Parallel tool calls run their hooks at once. Each draws its position in arrival order, under a lock held only
    # for the draw, and records itself as that position's holder.
    with open(os.path.join(changelog_dir, "seq.lock"), "a") as seq_lock:
        fcntl.flock(seq_lock, fcntl.LOCK_EX)
        if _epoch() != epoch:  # enable started another capture since this hook loaded its config
            return
        k = _read_int(seqf)
        _write_atomic(seqf, str(k + 1))
        _write_atomic(held % k, str(os.getpid()))
    local = os.path.join(changelog_dir, "%04d.tgz" % k)
    # The archives are incremental, so they're made in position order: wait for this position's turn, passing over
    # positions whose hook died first. A killed hook leaves a gap at its own position instead of shifting later ones.
    archived = stored = False
    while True:
        with open(os.path.join(changelog_dir, "lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if _epoch() != epoch:  # its capture was replaced; this position is no longer its to write
                return
            turn = _read_int(turnf)
            while turn < k and not _alive(changelog_dir, turn):
                turn += 1
            if turn == k:
                # tar works on a copy of the incremental state, promoted with the roots present only once the archive
                # is stored: a capture that isn't stored leaves its writes, and its removals, for the next to carry.
                work = "%s.%d" % (snar, k)
                if os.path.exists(snar):
                    shutil.copyfile(snar, work)
                present = [r for r in roots if os.path.exists(r)]
                removed = [r for r in _read_json(present_file, []) if r not in present]
                record = {"position": k, "session_id": session, "removed": removed}
                # The metadata lives outside the roots, so tar takes it as one more member unless a root holds it.
                written = bool(meta) and _write_meta(meta, k, record)
                members = [*present, meta] if written and cfg.get("meta_explicit") else present
                # A file the agent can't read is one it didn't write: skipped, so the rest of the increment is kept.
                tar = subprocess.run(
                    ["tar", "--ignore-failed-read", "--listed-incremental=" + work, "--exclude=" + changelog_dir,
                     "-czf", local, *members],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, pass_fds=(lock.fileno(),),
                )
                archived = True
                try:
                    if tar.returncode >= 2:
                        sys.stderr.write("[changelog] tar failed seq=%d rc=%d: %s\n" % (k, tar.returncode, _tail(tar.stderr)))
                    else:
                        if tar.returncode == TAR_FILE_CHANGED_RC:
                            sys.stderr.write("[changelog] tar warned seq=%d: %s\n" % (k, _tail(tar.stderr)))
                        stored = _store(changelog_dir, cfg, k, local)
                except Exception as e:
                    sys.stderr.write("[changelog] upload of seq=%d failed; the next capture carries its writes: %s\n"
                                     % (k, e))
                finally:
                    if stored:
                        os.replace(work, snar)
                        _write_atomic(present_file, json.dumps(present))
                    else:
                        _remove(work)
                    _remove(local)
            if turn >= k:
                _write_atomic(turnf, str(max(turn, k + 1)))
                break
        time.sleep(TURN_POLL_S)
    _remove(held % k)
    if not archived:  # a later position went first, so this one would replay out of order
        sys.stderr.write("[changelog] skipped seq=%d: a later position was archived first\n" % k)
    elif stored:
        sys.stderr.write("[changelog] captured seq=%d tool=%s -> %06d.tgz\n" % (k, tool, k))


if __name__ == "__main__":
    # The CLI stops a hook past its timeout with SIGTERM; exiting through SystemExit takes back its reservation.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(SIGTERM_EXIT_CODE))
    try:
        main()
    except Exception as e:
        sys.stderr.write("[changelog] hook error (non-fatal): %s\n" % e)
    sys.exit(0)
'''
_DEFAULT_CONFIG_LINE = 'CFG = "/tmp/changelog/config.json"'


def gnu_tar() -> bool:
    """Whether this host's tar is GNU tar, which the changelog's incremental archives need."""
    try:
        version = subprocess.run(["tar", "--version"], capture_output=True, text=True,
                                 timeout=_TAR_VERSION_TIMEOUT_S).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "GNU tar" in version


def changelog_dir(environ: Mapping[str, str]) -> Path:
    """This agent's changelog directory: one per port, since agents installed on one host listen on their own, and
    outside every capture root, so a capture never archives the changelog itself."""
    return Path(tempfile.gettempdir()) / f"{_DIR_PREFIX}{environ.get('A2A_PORT') or 'default'}"


def outermost(paths: Iterable[str]) -> list[str]:
    """``paths`` without duplicates or any inside another, so no file is archived twice."""
    kept: list[str] = []
    for path in dict.fromkeys(os.path.normpath(p) for p in paths):
        if not any(path == k or path.startswith(k.rstrip("/") + "/") for k in kept):
            kept = [k for k in kept if not k.startswith(path.rstrip("/") + "/")] + [path]
    return kept


def merge_settings_hook(path: Path, *, event: str, matcher: str, command: str, timeout: int) -> None:
    """Add a hook running ``command`` on ``event`` to a CLI's JSON settings file, once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        settings = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(settings, dict):
            settings = {}
    except Exception:
        settings = {}
    entries = settings.setdefault("hooks", {}).setdefault(event, [])
    already = any(
        isinstance(e, dict) and any(h.get("command") == command for h in e.get("hooks", []))
        for e in entries
    )
    if not already:
        entries.append({"matcher": matcher, "hooks": [{"type": "command", "command": command, "timeout": timeout}]})
    path.write_text(json.dumps(settings, indent=2))


def remove_settings_hooks(path: Path, *, event: str, matches: Callable[[str], bool]) -> bool:
    """Drop every ``event`` hook entry whose command ``matches``; whether the file changed."""
    if not path.exists():
        return False
    try:
        settings = json.loads(path.read_text())
    except Exception:
        return False
    hooks = settings.get("hooks") if isinstance(settings, dict) else None
    entries = hooks.get(event) if isinstance(hooks, dict) else None
    if not isinstance(entries, list):
        return False
    kept = [
        e for e in entries
        if not (isinstance(e, dict) and any(matches(str(h.get("command", ""))) for h in e.get("hooks", [])))
    ]
    if len(kept) == len(entries):
        return False
    hooks[event] = kept
    path.write_text(json.dumps(settings, indent=2))
    return True


def is_capture_command(command: str) -> bool:
    """Whether ``command`` runs some agent's capture script: a restored capture agent's hook names its own."""
    return f"/{_DIR_PREFIX}" in command and command.endswith("capture.py")


def is_missing_capture_command(command: str) -> bool:
    """Whether ``command`` runs a capture script this agent doesn't have: a hook restored from the agent that
    captured, rather than this agent's own."""
    return is_capture_command(command) and not Path(shlex.split(command)[-1]).exists()


def _stderr_tail(stderr: bytes) -> str:
    return stderr.decode(errors="replace").strip()[-_TAR_STDERR_TAIL_CHARS:]


def _own_directory(path: Path) -> Path:
    """``path`` made if missing, 0700; refused unless it's a real directory this process's user owns, since the
    capture's config holds the grant and its scripts run as the agent."""
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except FileExistsError:
        pass
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise HTTPException(status_code=500, detail=(
            f"changelog directory {path} must be a directory, not a link, owned by uid {os.geteuid()}"))
    path.chmod(0o700)
    return path


# Never removed on replay, whatever an increment's metadata says.
_SYSTEM_DIRS = frozenset({"/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib32", "/lib64", "/libx32",
                          "/media", "/mnt", "/opt", "/proc", "/root", "/run", "/sbin", "/srv", "/sys", "/tmp", "/usr",
                          "/var"})


def _under(path: str, roots: Iterable[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip("/") + "/") for root in roots)


def is_meta_member(member: str) -> bool:
    """Whether an archive member is a capture's own metadata rather than a file in the layer it restores."""
    return _META_MEMBER.search(member) is not None


def removable(path: object) -> bool:
    """Whether a removed root an increment names may be removed on replay: an absolute, normalized path that exists
    and isn't / or a top-level system directory."""
    return (isinstance(path, str) and os.path.isabs(path) and os.path.normpath(path) == path
            and path not in _SYSTEM_DIRS and os.path.lexists(path))


def _remove_tree(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)


class Changelog:
    """The ``enable-changelog`` / ``apply-changelog`` pair: ``install_hook`` registers the capture as the CLI's
    post-tool hook, ``after_apply`` runs once increments are extracted, and ``resume(target, session)`` binds a context
    to the restored conversation, or returns None."""

    def __init__(
        self,
        default_roots: Iterable[str],
        install_hook: Callable[[str], None],
        resume: Callable[[str | None, str | None], Awaitable[str | None]],
        directory: Path,
        after_apply: Callable[[], None],
    ) -> None:
        self.default_roots = list(default_roots)
        self.install_hook = install_hook
        self.resume = resume
        self.after_apply = after_apply
        self.directory = directory
        # Each capture's position, session and the roots it saw removed: beside the changelog, outside its roots.
        self.meta_dir = directory.parent / f"{directory.name}{META_DIR_SUFFIX}"
        self.meta_path = self.meta_dir / META_NAME
        self.snar = f"{directory}/state.snar"
        self.config_path = f"{directory}/config.json"
        self.capture_path = f"{directory}/capture.py"
        self.hook_command = f"{shlex.quote(sys.executable)} {shlex.quote(self.capture_path)}"

    def capture_script(self) -> str:
        """``CAPTURE_SCRIPT`` reading this changelog's config."""
        return CAPTURE_SCRIPT.replace(_DEFAULT_CONFIG_LINE, f"CFG = {json.dumps(self.config_path)}")

    async def enable(self, request: NamespaceChangelogEnableRequest) -> NamespaceChangelogEnableResponse:
        """Write the capture script and config, baseline the incremental tar so each increment is a pure delta, and
        install the hook. Idempotent."""
        roots = request.roots
        if roots is None:
            roots = list(self.default_roots)
        elif not roots:
            raise HTTPException(status_code=400, detail="roots must be a non-empty list of strings (omit to use defaults)")
        await asyncio.to_thread(self._reset, roots, request.write_namespace.model_dump(mode="json"))
        self.install_hook(self.hook_command)
        logger.info(f"agent-changelog configured: object-transfer grant roots={roots}")
        return NamespaceChangelogEnableResponse(roots=roots)

    def _reset(self, roots: list[str], write_namespace: dict) -> None:
        roots = [os.path.normpath(root) for root in roots]
        config = {"dir": str(self.directory), "snar": self.snar, "roots": roots, "epoch": uuid.uuid4().hex,
                  "write_namespace": write_namespace, "meta": str(self.meta_path),
                  "meta_explicit": not _under(str(self.meta_path), roots)}
        _own_directory(self.directory)
        _own_directory(self.meta_dir)
        # Reset under both capture locks, so a hook still running for an earlier capture can't write into this one:
        # it finds the new epoch in the config and stops.
        with open(self.directory / "lock", "a") as lock, open(self.directory / "seq.lock", "a") as seq_lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            fcntl.flock(seq_lock, fcntl.LOCK_EX)
            Path(self.capture_path).write_text(self.capture_script())
            Path(self.config_path).write_text(json.dumps(config))
            for stale in (
                self.directory / "seq",
                self.directory / "turn",
                self.directory / "namespace.json",
                Path(self.snar),
                self.meta_path,
                *self.directory.glob("held.*"),
                *self.directory.glob("state.snar.*"),
            ):
                stale.unlink(missing_ok=True)
            present = [r for r in roots if os.path.exists(r)]
            (self.directory / "present.json").write_text(json.dumps(present))
            subprocess.run(
                ["tar", "--ignore-failed-read", f"--listed-incremental={self.snar}", f"--exclude={self.directory}",
                 "-cf", "/dev/null", *present],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    async def apply(self, request: ObjectChangelogApplyRequest) -> ObjectChangelogApplyResponse:
        """Extract the granted increments in order at / onto a fresh agent of the same image, deletions included; an
        increment that isn't a readable archive fails the apply (422) before it's extracted."""
        applied, session = await self._apply_increments(request.increments)
        failed = [(sequence, rc) for sequence, rc in applied if rc]
        shown = ", ".join(f"{sequence:06d} rc={rc}" for sequence, rc in failed[:_FAILED_INCREMENTS_LOGGED])
        more = f" and {len(failed) - _FAILED_INCREMENTS_LOGGED} more" if len(failed) > _FAILED_INCREMENTS_LOGGED else ""
        logger.info(f"agent-changelog applied {len(applied)} increments"
                    + (f"; tar exited non-zero on {len(failed)}: {shown}{more}" if failed else ""))
        self.after_apply()
        context_id = await self.resume(request.target_context_id, session) if request.resume_conversation else None
        return ObjectChangelogApplyResponse(count=len(applied), context_id=context_id)

    @staticmethod
    def _metadata(local: Path, members: list[str], sequence: int) -> dict | None:
        """The increment's own metadata, read from the archive, or None when it carries none: one captured before
        captures wrote it, or whose metadata couldn't be written."""
        if not members:
            return None
        read = subprocess.run(["tar", "--incremental", "-xzOf", str(local), members[-1]],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            meta = json.loads(read.stdout) if read.returncode == 0 else None
        except ValueError:
            return None
        if not isinstance(meta, dict) or meta.get("position") != sequence:
            return None
        return meta

    def _replay_removals(self, sequence: int, removed: object) -> None:
        for path in removed if isinstance(removed, list) else []:
            if not removable(path):
                logger.warning(f"agent-changelog increment {sequence:06d}: not removing {path!r}")
                continue
            _remove_tree(path)
            logger.info(f"agent-changelog increment {sequence:06d}: removed root {path}")

    async def _apply_increments(self, increments: list[ChangelogIncrement]) -> tuple[list[tuple[int, int]], str | None]:
        """Each increment's sequence and tar's exit status, and the session of the last one's tool call, or None when
        it recorded none."""
        apply_dir = _own_directory(_own_directory(self.directory) / "apply")
        applied: list[tuple[int, int]] = []
        session: str | None = None
        for increment in increments:
            local = apply_dir / f"{increment.sequence:06d}.tgz"
            try:
                await download(increment.object, local)
                listing = await asyncio.to_thread(
                    subprocess.run, ["tar", "--incremental", "-tzf", str(local)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                if listing.returncode != 0:
                    raise HTTPException(
                        status_code=422,
                        detail=f"changelog increment {increment.sequence:06d} is not a readable archive: "
                        f"{_stderr_tail(listing.stderr)}",
                    )
                meta_members = [member for member in listing.stdout.decode(errors="replace").splitlines()
                                if is_meta_member(member)]
                meta = await asyncio.to_thread(self._metadata, local, meta_members, increment.sequence)
                # The metadata is the capture's record, not part of the layer it restores.
                excluded = [f"--exclude={member}" for member in meta_members]
                proc = await asyncio.to_thread(
                    subprocess.run,
                    ["tar", "--incremental", "--no-wildcards", *excluded, "-xzf", str(local), "-C", "/"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
            finally:
                local.unlink(missing_ok=True)
            if proc.returncode != 0:
                # Expected for members another user can't restore (e.g. "etc: Cannot utime"); the writable roots
                # still restore, so it counts.
                logger.warning(f"agent-changelog increment {increment.sequence:06d}: tar exited {proc.returncode}: "
                               f"{_stderr_tail(proc.stderr)}")
            if meta is not None:
                await asyncio.to_thread(self._replay_removals, increment.sequence, meta.get("removed"))
            recorded = meta.get("session_id") if meta is not None else None
            session = recorded if isinstance(recorded, str) and recorded else None
            applied.append((increment.sequence, proc.returncode))
        return applied, session
