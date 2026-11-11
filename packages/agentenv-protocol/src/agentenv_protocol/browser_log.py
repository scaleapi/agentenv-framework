"""A website browser's log: one entry for each action agents take on it, through CDP, MCP tools or pixels, and an
event each time a tab opens, navigates or closes, with what caused it. Its browser's server writes it, as JSON lines
with a sequence number each, in a file that lasts the computer's boot: a server that restarts carries on the sequence,
and a reset is a line, not a new file.

An entry is acting or reading by its CDP method or ``browser/v1`` op. An acting entry keeps its parameters with what
could carry a secret taken out: text typed while a password field has focus, cookie and header values, the values of
URL query keys that name a credential, and a request's body; a long value is cut, with its length and sha256. A
reading entry keeps none."""
from __future__ import annotations

import asyncio
import base64
import contextvars
import hashlib
import json
import logging
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

#: Where a browser's server writes its log: the computer's log folder, else its own folder when that can't be written.
LOG_DIR = Path("/var/log/agentenv")
#: The header a driver forwards an MCP tool call to the browser with: the tool, its call's id and its arguments.
TOOL_CALL_HEADER = "AgentEnv-Tool-Call"
#: The longest a string in an entry may be before it's cut, with its length and sha256 kept.
CUT_CHARS = 2048
#: The longest an error's message in an entry is kept.
ERROR_CHARS = 300
#: The longest the tool-call header may be; past it, its arguments are dropped.
HEADER_CHARS = 12_000
#: How many lines a read of the log returns at most.
READ_LIMIT = 5000
#: How long a tab event waits on the commands in flight that may have caused it.
CAUSE_WAIT_S = 5.0
#: How long after a reload or a move in history its tab's next navigation is taken as its.
NEXT_NAVIGATION_S = 10.0
REDACTED = "<redacted>"

#: The domain whose every command acts.
ACTING_DOMAINS = frozenset({"Input"})
#: Per domain, the method names that act on the browser or a page, by their start; any other method reads.
ACTING_METHODS: dict[str, tuple[str, ...]] = {
    "Page": ("navigate", "reload", "stopLoading", "close", "bringToFront", "handleJavaScriptDialog",
             "setDocumentContent", "addScriptToEvaluateOnNewDocument", "removeScriptToEvaluateOnNewDocument",
             "setBypassCSP", "setGeolocationOverride", "setWebLifecycleState", "crash", "resetNavigationHistory"),
    "Runtime": ("evaluate", "callFunctionOn", "runScript", "addBinding"),
    "Target": ("createTarget", "closeTarget", "activateTarget", "createBrowserContext", "disposeBrowserContext"),
    "Browser": ("setWindowBounds", "setDownloadBehavior", "grantPermissions", "resetPermissions", "setPermission",
                "cancelDownload", "close", "crash"),
    "DOM": ("set", "remove", "move", "copy", "undo", "redo", "focus", "scrollIntoView"),
    "CSS": ("set", "add", "create"),
    "Emulation": ("set", "clear", "reset"),
    "Fetch": ("fulfill", "continue", "fail"),
    "Network": ("set", "clear", "delete", "emulate", "replay"),
    "Storage": ("set", "clear", "delete", "reset", "override"),
    "DOMStorage": ("set", "remove", "clear"),
    "IndexedDB": ("clear", "delete"),
    "CacheStorage": ("delete",),
    "ServiceWorker": ("unregister", "skipWaiting", "stopWorker", "stopAllWorkers", "dispatch", "deliverPushMessage"),
}
#: The ``browser/v1`` ops that act; any other reads.
ACTING_OPS = frozenset({"page/input", "navigate", "history", "tabs/open", "tabs/activate", "tabs/close",
                        "dialog/answer", "files/choose", "cookies/set", "cookies/delete", "local_storage/set",
                        "local_storage/clear", "sites/set"})
#: The ``browser/v1`` ops an agent's calls are never logged for: the log itself, and agentenv's own bookkeeping.
UNLOGGED_OPS = frozenset({"log", "record"})
#: The CDP methods and ops whose text is typed into the page.
TYPING_METHODS = frozenset({"Input.insertText", "Input.dispatchKeyEvent", "Input.imeSetComposition"})
#: What a key event says of the key it types, all of it dropped when a password field has focus.
KEY_FIELDS = ("text", "unmodifiedText", "key", "code", "keyIdentifier", "windowsVirtualKeyCode",
              "nativeVirtualKeyCode", "commands")
#: The words of a URL query key that name a credential, so its value is taken out.
SECRET_WORDS = frozenset({"token", "key", "apikey", "secret", "password", "passwd", "pwd", "auth", "authorization",
                          "code", "session", "sessionid", "sid", "sig", "signature", "credential", "credentials", "jwt",
                          "ticket", "assertion", "otp", "saml", "samlresponse", "nonce"})
#: The query keys whose every value is taken out, by their start: signed-URL parameters.
SECRET_PREFIXES = ("x-amz-", "x-goog-", "x-ms-")
_WORDS = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")

#: The action under way in this task: the entry an op of the browser's server or a relayed command is, which the
#: browser's control credits the tab events it causes to, and the tab it acted on.
ACTOR: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("agentenv_browser_actor", default=None)


def acts(method: str) -> bool:
    """Whether the CDP ``method`` acts on the browser or a page, rather than reading it."""
    domain, _, name = method.partition(".")
    return domain in ACTING_DOMAINS or name.startswith(ACTING_METHODS.get(domain, ("\0",)))


def op_acts(op: str) -> bool:
    """Whether the ``browser/v1`` ``op`` acts on the browser or a page, rather than reading it."""
    return op in ACTING_OPS


def cut(value: Any) -> Any:
    """``value`` with each string longer than ``CUT_CHARS`` cut to that, its length and sha256 beside it."""
    if isinstance(value, str):
        if len(value) <= CUT_CHARS:
            return value
        return {"cut": value[:CUT_CHARS], "length": len(value), "sha256": hashlib.sha256(value.encode()).hexdigest()}
    if isinstance(value, Mapping):
        return {key: cut(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cut(item) for item in value]
    return value


def withheld(value: Any) -> dict:
    """What an entry keeps of a value it can't hold: its length and sha256."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return {"redacted": True, "length": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def redact_url(url: Any) -> Any:
    """``url`` with its password and the values of its query keys, and of a query-like fragment, that name a credential
    taken out; its scheme, host, path and other query values stay."""
    if not isinstance(url, str) or "://" not in url:
        return url
    try:
        parts = urlsplit(url)
        netloc = parts.netloc
        if "@" in netloc:
            userinfo, _, host = netloc.rpartition("@")
            netloc = f"{userinfo.partition(':')[0]}:{REDACTED}@{host}" if ":" in userinfo else netloc
        return urlunsplit((parts.scheme, netloc, parts.path, _redact_query(parts.query),
                           _redact_query(parts.fragment) if "=" in parts.fragment else parts.fragment))
    except ValueError:
        return url


def _redact_query(query: str) -> str:
    if not query:
        return query
    pairs = parse_qsl(query, keep_blank_values=True)
    if not pairs:
        return query
    return "&".join(f"{quote(key, safe='')}={REDACTED if _secret_key(key) else quote(value, safe='')}"
                    for key, value in pairs)


def _secret_key(key: str) -> bool:
    lowered = key.lower()
    return lowered.startswith(SECRET_PREFIXES) or bool(SECRET_WORDS & {word.lower() for word in _WORDS.findall(key)})


def redacted_params(method: str, params: dict, *, password: bool = False) -> dict:
    """The parameters of an acting CDP ``method`` as its entry keeps them: what could carry a secret taken out, and
    with a password field focused, what it types; then cut."""
    kept = dict(params)
    if method in ("Network.setCookie",) and "value" in kept:
        kept["value"] = REDACTED
    if method in ("Network.setCookies", "Storage.setCookies") and isinstance(kept.get("cookies"), list):
        kept["cookies"] = [{**cookie, "value": REDACTED} if isinstance(cookie, dict) else cookie
                           for cookie in kept["cookies"]]
    if method == "Network.setExtraHTTPHeaders" and isinstance(kept.get("headers"), dict):
        kept["headers"] = {name: REDACTED for name in kept["headers"]}
    for field in ("headers", "responseHeaders"):
        if method.startswith("Fetch.") and isinstance(kept.get(field), list):
            kept[field] = [{**header, "value": REDACTED} if isinstance(header, dict) else header
                           for header in kept[field]]
    if method.startswith("Fetch.") and "binaryResponseHeaders" in kept:
        kept["binaryResponseHeaders"] = REDACTED
    if method.startswith("Fetch.") and "postData" in kept:
        kept["postData"] = withheld(kept["postData"])
    if method == "Fetch.continueWithAuth" and isinstance(kept.get("authChallengeResponse"), dict):
        kept["authChallengeResponse"] = {key: (REDACTED if key in ("username", "password") else value)
                                         for key, value in kept["authChallengeResponse"].items()}
    for field in ("url", "urls"):
        if field in kept:
            kept[field] = [redact_url(url) for url in kept[field]] if isinstance(kept[field], list) \
                else redact_url(kept[field])
    if password and method in TYPING_METHODS:
        kept = {key: value for key, value in kept.items() if key not in KEY_FIELDS}
        if "text" in params:
            kept["text"] = {"length": len(params["text"]) if isinstance(params["text"], str) else 0}
    return cut(kept)


def redacted_op(op: str, params: dict, *, password: bool = False) -> dict:
    """The parameters of an acting ``browser/v1`` ``op`` as its entry keeps them, as ``redacted_params`` keeps a CDP
    method's: page input event by event."""
    kept = dict(params)
    if op == "page/input" and isinstance(kept.get("events"), list):
        kept["events"] = [{**event, "params": redacted_params(str(event.get("method")), event.get("params") or {},
                                                              password=password)}
                          if isinstance(event, dict) else event for event in kept["events"]]
    if op == "cookies/set" and isinstance(kept.get("cookies"), list):
        kept["cookies"] = [{**cookie, "value": REDACTED} if isinstance(cookie, dict) else cookie
                           for cookie in kept["cookies"]]
    if "url" in kept:
        kept["url"] = redact_url(kept["url"])
    return cut(kept)


def redacted_args(tool: str, args: dict, *, password: bool = False) -> dict:
    """An MCP tool call's arguments as its entry keeps them: URLs as ``redact_url`` keeps them, and typed text, while
    a password field has focus, as its length."""
    kept = {key: redact_url(value) if key == "url" else value for key, value in args.items()}
    if password and isinstance(kept.get("text"), str):
        kept["text"] = {"length": len(kept["text"])}
    return cut(kept)


def tool_call_header(tool: str, call: str, args: Mapping[str, Any]) -> str:
    """The header a driver forwards a tool call with: the tool, its call's id and its arguments, cut; without them when
    even cut they'd make it too long."""
    body = {"tool": tool, "call": call, "args": cut(args)}
    encoded = base64.urlsafe_b64encode(json.dumps(body).encode()).decode()
    if len(encoded) > HEADER_CHARS:
        body["args"] = {key: withheld(cut(value)) for key, value in args.items()}
        encoded = base64.urlsafe_b64encode(json.dumps(body).encode()).decode()
    if len(encoded) > HEADER_CHARS:
        encoded = base64.urlsafe_b64encode(json.dumps({"tool": tool, "call": call, "args": {}}).encode()).decode()
    return encoded


def read_tool_call(value: Optional[str]) -> Optional[dict]:
    """The tool call a driver forwarded, from its header; None without one or for one it can't read."""
    if not value:
        return None
    try:
        body = json.loads(base64.urlsafe_b64decode(value.encode()))
    except (ValueError, TypeError):
        return None
    if not isinstance(body, dict) or not isinstance(body.get("tool"), str):
        return None
    return {"tool": body["tool"], "call": str(body.get("call") or ""),
            "args": body["args"] if isinstance(body.get("args"), dict) else {}}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class BrowserLog:
    """The log in ``path``: each line a JSON object with its ``seq`` and its time, ``at``, and one of ``action``,
    ``tab`` or ``reset``. A line's sequence number is taken when what it records begins, and the line is written once
    it ends, so lines may be written out of order; a read returns them in order, up to the last whose every earlier
    line is written. The file is opened at the first line, after any a server before this one wrote."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._seq, self._partial = _last_seq(path)
        self._pending: set[int] = set()
        self._file: Any = None
        self._unwritable = False
        #: Per tab, the sequence number of the latest acting entry on it.
        self.last_acting: dict[str, int] = {}

    @classmethod
    def for_app(cls, name: str, fallback: Path) -> BrowserLog:
        """The log of the browser app ``name``: in the computer's log folder, else in ``fallback``, its own folder."""
        folder = LOG_DIR if os.access(LOG_DIR, os.W_OK) else fallback
        return cls(folder / f"browser-{name}.jsonl")

    def reserve(self) -> int:
        """The next sequence number, held until its line is written."""
        self._seq += 1
        self._pending.add(self._seq)
        return self._seq

    def write(self, seq: int, line: dict, at: Optional[str] = None) -> None:
        """The line held as ``seq`` written, with its time."""
        self._pending.discard(seq)
        if self._file is None and not self._open():
            return
        try:
            self._file.write(json.dumps({"seq": seq, "at": at or _now(), **line}, default=str) + "\n")
            self._file.flush()
        except (OSError, ValueError) as e:
            logger.warning(f"the browser's log couldn't take line {seq}: {type(e).__name__}: {e}")

    def _open(self) -> bool:
        """The file open for appending, on a new line; False, once warned, when it can't be."""
        if self._unwritable:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a", encoding="utf-8")
            if self._partial:
                self._file.write("\n")
            return True
        except OSError as e:
            logger.warning(f"the browser's log can't be written at {self.path}: {type(e).__name__}: {e}")
            self._unwritable = True
            return False

    def begin(self, action: dict) -> dict:
        """An action that begins now, held until ``finish``: its sequence number, time and fields so far. An acting one
        on a tab becomes that tab's latest."""
        seq = self.reserve()
        if action.get("acts") and action.get("tab"):
            self.last_acting[action["tab"]] = seq
        return {"seq": seq, "at": _now(), "action": action}

    def finish(self, begun: dict, **fields: Any) -> None:
        """The action ``begun`` written, with what it ended with."""
        action = {**begun["action"], **fields}
        if action.get("acts") and action.get("tab") and self.last_acting.get(action["tab"], 0) < begun["seq"]:
            self.last_acting[action["tab"]] = begun["seq"]
        self.write(begun["seq"], {"action": action}, at=begun["at"])

    def note(self, kind: str, body: dict) -> int:
        """A line of ``kind`` written now: a ``tab`` event, or a ``reset``."""
        seq = self.reserve()
        self.write(seq, {kind: body})
        return seq

    def read(self, after: int = 0, limit: int = READ_LIMIT) -> dict:
        """The lines after sequence number ``after``, in order, up to ``limit`` of them and to the last whose every
        earlier line is written: ``lines``, ``last``, the sequence number of that last one, and ``more`` when the limit
        left some."""
        complete = min(self._pending) - 1 if self._pending else self._seq
        lines = []
        try:
            with self.path.open(encoding="utf-8") as file:
                for raw in file:
                    try:
                        line = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(line, dict) and isinstance(line.get("seq"), int) and after < line["seq"] <= complete:
                        lines.append(line)
        except OSError:
            pass
        lines.sort(key=lambda line: line["seq"])
        more = len(lines) > limit
        lines = lines[:limit]
        last = (lines[-1]["seq"] if lines else after) if more else max(after, complete)
        return {"lines": lines, "last": last, "more": more}

    def complete_text(self) -> str:
        """Every line written whose every earlier line is, in order, as the file's text."""
        read = self.read(0, limit=1 << 30)
        return "".join(json.dumps(line, default=str) + "\n" for line in read["lines"])


def _last_seq(path: Path) -> tuple[int, bool]:
    """The highest sequence number in the log at ``path``, and whether its last line was left partial."""
    try:
        data = path.read_bytes()
    except OSError:
        return 0, False
    highest = 0
    for raw in data.splitlines():
        try:
            line = json.loads(raw)
        except ValueError:
            continue
        if isinstance(line, dict) and isinstance(line.get("seq"), int):
            highest = max(highest, line["seq"])
    return highest, bool(data) and not data.endswith(b"\n")


def caused(cause: Optional[dict], log: BrowserLog, tab: Optional[str]) -> dict:
    """What a tab event says caused it: the command it was, by its entry, ``via`` and ``who``; agentenv itself; or,
    with no ``cause``, the page, after the latest acting entry on ``tab`` if there is one."""
    if cause is None:
        after = log.last_acting.get(tab) if tab else None
        return {"by": "page", **({"after": after} if after else {})}
    if cause.get("via") == "agentenv":
        return {"by": "agentenv"}
    return {"by": "command", "entry": cause["seq"], "via": cause["via"], "who": cause.get("who")}


def log_time() -> str:
    """Now, as the log writes times."""
    return _now()


class Causes:
    """The commands in flight that may open, close or navigate a tab, so the tab event each causes can name it: a new
    tab's command answers after the browser says the tab opened, and a navigation's may too."""

    def __init__(self) -> None:
        self._creating: set[asyncio.Future] = set()
        self._created: dict[str, dict] = {}
        self._closing: dict[str, dict] = {}
        self._navigating: dict[str, set[asyncio.Future]] = {}
        self._loaders: dict[str, dict] = {}
        self._next: dict[str, tuple[dict, float]] = {}

    def creating(self, answer: asyncio.Future, cause: dict) -> None:
        """A ``Target.createTarget`` in flight, ``answer`` its answer's target id, sent by ``cause``: once it lands, the
        new tab's first navigation, to the URL it was opened at, is taken as ``cause``'s too."""
        self._creating.add(answer)
        answer.add_done_callback(lambda done: self._landed(done, self._created, cause, self._creating))
        answer.add_done_callback(lambda done: self._first_navigation(done, cause))

    def _first_navigation(self, done: asyncio.Future, cause: dict) -> None:
        if not done.cancelled() and done.exception() is None and isinstance(done.result(), str):
            self._next.setdefault(done.result(), (cause, asyncio.get_running_loop().time() + NEXT_NAVIGATION_S))

    def closing(self, target: str, cause: dict) -> None:
        self._closing[target] = cause

    def navigating(self, target: str, answer: asyncio.Future, cause: dict) -> None:
        """A ``Page.navigate`` of ``target`` in flight, ``answer`` its answer's loader id, sent by ``cause``."""
        waiting = self._navigating.setdefault(target, set())
        waiting.add(answer)
        answer.add_done_callback(lambda done: self._landed(done, self._loaders, cause, waiting))

    def navigates_next(self, target: str, cause: dict) -> None:
        """A reload or a move in history of ``target``, whose answer names no loader: its next navigation is taken as
        its, within ``NEXT_NAVIGATION_S``."""
        self._next[target] = (cause, asyncio.get_running_loop().time() + NEXT_NAVIGATION_S)

    @staticmethod
    def _landed(done: asyncio.Future, into: dict, cause: dict, waiting: set) -> None:
        waiting.discard(done)
        if not done.cancelled() and done.exception() is None and isinstance(done.result(), str):
            into[done.result()] = cause

    async def opened(self, target: str) -> Optional[dict]:
        """What opened ``target``, once the creates in flight have answered."""
        if target not in self._created and self._creating:
            await asyncio.wait(set(self._creating), timeout=CAUSE_WAIT_S)
        return self._created.pop(target, None)

    def closed(self, target: str) -> Optional[dict]:
        return self._closing.pop(target, None)

    async def navigated(self, target: str, loader: Optional[str]) -> Optional[dict]:
        """What navigated ``target`` to ``loader``, once its navigations in flight have answered."""
        if loader and loader not in self._loaders and self._navigating.get(target):
            await asyncio.wait(set(self._navigating[target]), timeout=CAUSE_WAIT_S)
        pending = self._next.pop(target, None)
        if loader and (cause := self._loaders.pop(loader, None)) is not None:
            return cause
        if pending is not None and pending[1] >= asyncio.get_running_loop().time():
            return pending[0]
        return None
