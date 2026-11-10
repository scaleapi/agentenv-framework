"""An app's data plane as its own server serves it (``AppDataPlane``), by the app's data hooks or, with none, the
defaults on the paths it keeps its data in, under the session user's home: a load writes the files it's given under
them, an export returns the files there, and a reset puts back the copy its server kept when it first started on this
boot, which is the app as installed, since every boot starts from the computer's image. An export copies the files with
the app paused, or, where the app holds one so that no other process may read it, as Windows lets an app hold a file
and a pause doesn't release, with it stopped and relaunched after."""
from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Optional

from ..agent_env_environment import OP_ADD_DATA, OP_GET_DATA, OP_RESET_DATA
from ..transfers import WriteNamespaceGrant
from ..types import METHOD_ADD, METHOD_GET, METHOD_RESET, FilePart, uploaded_object_path
from .app_runtime import paused_app
from .dataplane import data_files, file_parts, part_bytes

if TYPE_CHECKING:
    from .app_runtime import AppRuntime

logger = logging.getLogger(__name__)

#: The cause a launch after each data call logs.
_CAUSES = {OP_ADD_DATA: METHOD_ADD, OP_GET_DATA: METHOD_GET, OP_RESET_DATA: METHOD_RESET}
#: Where an app's server keeps its paths as installed, in the app's folder.
INITIAL = Path(".agentenv/initial")
#: Windows' errors for a file another process holds so that no other may read it: a sharing violation, and a lock on
#: the part read.
_HELD = (32, 33)


class HeldByApp(OSError):
    """A file under the app's paths that the running app holds so that no other process may read it, which stopping
    the app releases and pausing it doesn't."""


class AppDataPlane:
    """An app's data plane, served around its contract: one data call or launch at a time; a load and a reset stop the
    running app first and relaunch it after, but for those the app serves live, and a load that fails is followed by a
    reset, when the app says so; an export the app holds a file of (``HeldByApp``) stops it and relaunches it too. A
    call fails once it runs
    past ``timeout_s`` and is cancelled: what awaits stops at once, but a thread can't be stopped, so a sync hook's or
    a default's thread is waited for, as are the reset and relaunch after it and the next call."""

    def __init__(self, runtime: AppRuntime, hooks: dict[str, Callable], paths: list[str], timeout_s: float, *,
                 live: frozenset[str] = frozenset(), keeps_paths: bool = False,
                 resets_failed_add: Callable[[], bool] = lambda: True,
                 relaunched: Optional[Callable[[str], Awaitable[None]]] = None) -> None:
        self.runtime = runtime
        self.hooks = hooks
        self.paths = paths
        self.timeout_s = timeout_s
        #: The operations served on the running app; whether a hook's reset needs the paths as installed; and whether
        #: a load that fails is followed by a reset now.
        self.live, self.keeps_paths, self.resets_failed_add = live, keeps_paths, resets_failed_add
        #: What a call that stopped the app awaits once it has launched it again, whose error fails the call.
        self.relaunched = relaunched
        self.turn = asyncio.Lock()
        self.changing: set[asyncio.Task] = set()

    def keep_initial(self) -> None:
        """Keep the app's paths as installed, before anything changes them, when they serve its data or its own reset
        puts them back."""
        if self.paths and (not self.hooks or self.keeps_paths):
            keep_initial(self.paths)

    async def call(self, op: str, *args: Any) -> Any:
        """``op``'s answer, by its hook or its paths' default, once the data call before has ended. The change it
        makes, with the reset and relaunch after it, holds the turn until it ends, past an answer given at its
        timeout."""
        served = self._served(op)
        await self._wait_turn(op)
        answer = asyncio.get_running_loop().create_future()
        change = asyncio.ensure_future(self._change(op, served, args, answer))
        self.changing.add(change)
        change.add_done_callback(self.changing.discard)
        change.add_done_callback(lambda _: self.turn.release())
        return await answer

    async def launch(self) -> int:
        """Launch the app once the data call before has ended; its pid."""
        await self._wait_turn("launch")
        try:
            return await self.runtime.launch_app()
        finally:
            self.turn.release()

    async def _change(self, op: str, served: Callable, args: tuple, answer: asyncio.Future) -> None:
        """``op`` between the app's stop and its relaunch, answered on ``answer``: as it ends, or once it runs past
        ``timeout_s``. A live one runs on the running app, which a reset after it fails stops and relaunches; an app
        it ran on is running again after it fails, though the op stopped it."""
        try:
            running = op != OP_GET_DATA and op not in self.live and await self.runtime.stop_app()
            ran = op in self.live and self.runtime.running()
            try:
                result = await self._within_held(op, served, args, answer)
            except Exception:
                if op == OP_ADD_DATA and self.resets_failed_add():
                    if op in self.live:
                        running = await self.runtime.stop_app() or ran
                    await self._undo(answer)
                elif op in self.live:
                    running = ran and not self.runtime.running()
                raise
            finally:
                if running:
                    await self.runtime.launch_app(_CAUSES[op])
            if running and self.relaunched is not None:
                await self.relaunched(op)
        except Exception as e:
            if not answer.done():
                answer.set_exception(e)
        else:
            answer.set_result(result)

    async def _within_held(self, op: str, served: Callable, args: tuple, answer: asyncio.Future) -> Any:
        """``served``'s result within ``timeout_s``; where the app holds a file of its paths (``HeldByApp``), again with
        the app stopped, in what's left of it, and the app relaunched after, as a load's is."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_s
        try:
            return await self._within(op, served, args, answer)
        except HeldByApp:
            stopped = await self.runtime.stop_app()
        try:
            result = await self._within(op, served, args, answer, deadline - loop.time())
        finally:
            if stopped:
                await self.runtime.launch_app(_CAUSES[op])
        if stopped and self.relaunched is not None:
            await self.relaunched(op)
        return result

    async def _undo(self, answer: asyncio.Future) -> None:
        """Reset the app's data after a load that failed."""
        if (reset := self._served(OP_RESET_DATA)) is None:
            return
        try:
            await self._within(OP_RESET_DATA, reset, (), answer)
        except Exception as e:
            logger.warning(f"the reset after a failed data/add failed too: {type(e).__name__}: {e}")

    async def _wait_turn(self, op: str) -> None:
        """Wait, within ``timeout_s``, for the data call before to end, its reset and relaunch included."""
        try:
            await asyncio.wait_for(self.turn.acquire(), self.timeout_s)
        except TimeoutError:
            raise TimeoutError(f"{op} waited past the app's data_timeout_s, {self.timeout_s:g} s, for the data call "
                               "before it to end") from None

    def _served(self, op: str) -> Optional[Callable]:
        """What serves ``op``: the app's hook, or its paths' default; None when nothing does."""
        if self.hooks:
            hook = self.hooks.get(op)
            if hook is not None and op == OP_GET_DATA and "write_namespace" not in inspect.signature(hook).parameters:
                return lambda write_namespace=None: hook()
            return hook
        if not self.paths:
            return None
        return {OP_ADD_DATA: functools.partial(add, paths=self.paths),
                OP_GET_DATA: functools.partial(get, self.paths),
                OP_RESET_DATA: functools.partial(reset, self.paths)}[op]

    async def _within(self, op: str, served: Callable, args: tuple, answer: asyncio.Future,
                      left_s: Optional[float] = None) -> Any:
        """``served``'s result within ``left_s``, ``timeout_s`` by default; past it, ``answer`` fails at once and the
        call is cancelled, and this raises the same once it has ended: at once for what awaits, after any thread it
        runs."""
        work = asyncio.ensure_future(_called(served, *args))
        try:
            return await asyncio.wait_for(asyncio.shield(work), max(0.0, self.timeout_s if left_s is None else left_s))
        except TimeoutError:
            late = TimeoutError(f"{op} ran past the app's data_timeout_s, {self.timeout_s:g} s")
            if not answer.done():
                answer.set_exception(late)
            work.cancel()
            await asyncio.wait([work])
            if not work.cancelled():
                work.exception()
            raise late from None


def keep_initial(paths: list[str]) -> None:
    """Copy what's under each of ``paths`` of the home into ``INITIAL``, unless this boot kept a copy already."""
    if INITIAL.exists():
        return
    staging = INITIAL.with_name(f"{INITIAL.name}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for path in paths:
        _copy(Path.home() / path, staging / path)
    staging.rename(INITIAL)


async def add(parts: list, paths: list[str]) -> None:
    """Write each file part at the path it names, which must be under one of ``paths``, through no link that leads
    out of them; nothing is written when one isn't."""
    files = {}
    for part in parts:
        path = uploaded_object_path(part) if isinstance(part, FilePart) else None
        if path is None:
            raise ValueError(f"data/add on this app takes file parts naming their path, got a {part.kind} part")
        if not any(PurePosixPath(path).is_relative_to(owned) for owned in paths):
            raise ValueError(f"{path!r} is under none of this app's paths: {', '.join(paths)}")
        files[path] = await part_bytes(part)
    await in_thread(write_within, Path.home(), files, paths)


async def get(paths: list[str], write_namespace: Optional[WriteNamespaceGrant]) -> list:
    """A part for each file under ``paths``, copied with the app this server launched paused (``paused_app``), so they
    are of one moment, and sent from the copy once it runs again; an app's own data/get hook calls it so too. A file
    the paused app holds raises ``HeldByApp``, for its data plane to copy them with it stopped."""
    INITIAL.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=INITIAL.parent) as staging:
        try:
            with paused_app():
                await in_thread(_stage, Path.home(), paths, Path(staging))
        except OSError as e:
            if getattr(e, "winerror", None) in _HELD:
                raise HeldByApp(f"the running app holds {e.filename or 'a file'} so that nothing else may read it") from e
            raise
        return await file_parts(Path(staging), paths, write_namespace)


async def reset(paths: list[str]) -> None:
    """Put each of ``paths`` back as ``keep_initial`` kept it, or remove it where it wasn't there then."""
    await in_thread(_restore, Path.home(), paths)


async def _called(fn: Callable, *args: Any) -> Any:
    """``fn``'s result, a sync one's from a thread, so it can't hold up the server."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    result = await in_thread(fn, *args)
    return await result if inspect.isawaitable(result) else result


async def in_thread(fn: Callable, *args: Any) -> Any:
    """``fn``'s result from a thread, which a cancelled call still waits for to the end, since a thread can't be
    stopped."""
    work = asyncio.ensure_future(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        await asyncio.wait([work])
        raise


def write_within(home: Path, files: dict[str, bytes], paths: list[str]) -> None:
    """Each file at its path under the home, none written when one leads out of ``paths`` through a link."""
    owned = [(home / path).resolve() for path in paths]
    if linked := [path for path in files if not any((home / path).resolve().is_relative_to(o) for o in owned)]:
        raise ValueError(f"{', '.join(map(repr, linked))} lead out of this app's paths through a link")
    for path, data in files.items():
        target = home / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _restore(home: Path, paths: list[str]) -> None:
    for path in paths:
        target = home / path
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
        _copy(INITIAL / path, target)


def _stage(home: Path, paths: list[str], staging: Path) -> None:
    """A copy in ``staging`` of each file an export of ``paths`` sends, by its path: a link's target, inside the home, as
    a file, and no socket or pipe."""
    for relative, source in data_files(home, paths):
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            shutil.copy2(source, target)


def _copy(source: Path, target: Path) -> None:
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(source, target, symlinks=True)
    elif source.exists() or source.is_symlink():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=False)
