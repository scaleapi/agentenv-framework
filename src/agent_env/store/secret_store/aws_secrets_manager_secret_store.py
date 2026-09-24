"""AWS Secrets Manager secret store backend."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Iterator, ItemsView, KeysView, Mapping, ValuesView
from typing import Any, Optional

import boto3
import yaml
from botocore.config import Config as BotocoreConfig

from agent_env.store.secret_store.secret_store import SecretStore

logger = logging.getLogger(__name__)

# GetSecretValue is $0.05/10k calls: a handful of pods on a 5-minute TTL is cents per month.
_DEFAULT_TTL_SECONDS = 300.0

# Floor between forced refresh() calls; doubles as the retry backoff after a failed re-fetch.
_DEFAULT_MIN_REFRESH_INTERVAL = 10.0

# Re-fetches run on the read path: fail in seconds, not botocore's default 60s x retries.
_BOTO_CONFIG = BotocoreConfig(
    connect_timeout=5, read_timeout=10, retries={"max_attempts": 3, "mode": "standard"}
)


class _BundleView(Mapping):
    """Live read-only view of the store's bundle: consumers that memoize ``_load()``'s
    return for the process lifetime (``Config._get_secret()`` does) still observe TTL
    re-fetches. Accessors bind to one snapshot each; not json/pickle/deepcopy-serializable.
    """

    def __init__(self, store: "AwsSecretsManagerSecretStore") -> None:
        self._store = store

    def __getitem__(self, key: str) -> Any:
        return self._store._current()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store._current())

    def __len__(self) -> int:
        return len(self._store._current())

    def get(self, key: str, default: Any = None) -> Any:
        return self._store._current().get(key, default)

    def keys(self) -> KeysView:
        return self._store._current().keys()

    def items(self) -> ItemsView:
        return self._store._current().items()

    def values(self) -> ValuesView:
        return self._store._current().values()

    def __eq__(self, other: Any) -> bool:
        return self._store._current() == other

    def __repr__(self) -> str:  # never render values
        return f"<_BundleView of {self._store._secret_name!r}>"


#: Shared by every caller, because the loggers being changed are process-global.
_suppress_guard = threading.Lock()
_suppress_depth = 0
_suppress_saved: list = []


@contextlib.contextmanager
def suppress_aws_body_logging():
    """Keep an AWS response body out of the logs for the duration of a call.

    botocore logs full HTTP request and response bodies at DEBUG. For most calls
    that is a useful trace; for ``GetSecretValue`` the response body *is* the
    secret, so any process that turns on DEBUG logging — a developer debugging an
    unrelated timeout, a verbose CI job — writes the entire bundle to stdout in
    plaintext. That has happened.

    Raising the level only around the fetch is deliberate: it does not fight the
    caller's logging config, and it does not silence botocore anywhere else. The
    previous level is always restored, including on the error path, because the
    exception carrying us out may itself need to be debugged.

    Reference-counted, because the loggers are process-global and the fetches
    are not serialised against each other — the secret store guards itself with
    its own lock, and any other caller reaching Secrets Manager directly is a
    different object with a different lock. Without the count, two overlapping
    fetches restore DEBUG when the *first* finishes, and the second one's
    ``SecretString`` goes straight to the handlers. Only the outermost context
    restores, and only it captured the levels worth restoring to.
    """
    global _suppress_depth, _suppress_saved

    loggers = [logging.getLogger(name) for name in
               ("botocore.parsers", "botocore.endpoint", "botocore.httpsession",
                "botocore.awsrequest", "urllib3.connectionpool")]

    with _suppress_guard:
        if _suppress_depth == 0:
            _suppress_saved = [(lg, lg.level) for lg in loggers]
            for lg in loggers:
                if lg.level < logging.INFO:
                    lg.setLevel(logging.INFO)
        _suppress_depth += 1
    try:
        yield
    finally:
        with _suppress_guard:
            _suppress_depth -= 1
            if _suppress_depth == 0:
                for lg, level in _suppress_saved:
                    lg.setLevel(level)
                _suppress_saved = []


class AwsSecretsManagerSecretStore(SecretStore):
    """Reads secrets from one AWS Secrets Manager secret holding a YAML/JSON mapping
    (a combined bundle secret).

    The bundle is cached (thread-safe) and re-fetched once older than ``ttl_seconds``
    (0 = cache for the process lifetime); ``refresh()`` forces a re-fetch, rate-limited
    to one per ``min_refresh_interval``. A failed TTL re-fetch serves the last-good
    bundle; only the first-ever fetch and an explicit ``refresh()`` raise.
    """

    def __init__(
        self,
        secret_name: str,
        region_name: str,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        min_refresh_interval: float = _DEFAULT_MIN_REFRESH_INTERVAL,
    ) -> None:
        self._secret_name = secret_name
        self._region_name = region_name
        self._ttl_seconds = float(ttl_seconds)
        self._min_refresh_interval = float(min_refresh_interval)
        self._values: Optional[dict] = None
        self._fetched_at: Optional[float] = None
        self._last_forced_refresh_at: Optional[float] = None
        self._last_failed_fetch_at: Optional[float] = None
        self._lock = threading.Lock()
        self._view = _BundleView(self)

    @classmethod
    def from_config(
        cls,
        *,
        secret_name: str,
        region: str,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        min_refresh_interval: float = _DEFAULT_MIN_REFRESH_INTERVAL,
    ) -> AwsSecretsManagerSecretStore:
        """Build from a ``[stores.secret.config]`` table (literal values only)."""
        return cls(
            secret_name,
            region,
            ttl_seconds=ttl_seconds,
            min_refresh_interval=min_refresh_interval,
        )

    def _fetch(self) -> dict:
        """Fetch and parse the bundle. Caller holds ``self._lock``."""
        client = boto3.client(
            "secretsmanager", region_name=self._region_name, config=_BOTO_CONFIG
        )
        with suppress_aws_body_logging():
            raw = client.get_secret_value(SecretId=self._secret_name)["SecretString"]
        try:
            loaded = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:
            # YAML error marks embed the raw secret line: re-raise sanitized, context dropped.
            mark = getattr(exc, "problem_mark", None)
            where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
            raise ValueError(
                f"AWS secret {self._secret_name!r} is not valid YAML/JSON{where}"
            ) from None
        if not isinstance(loaded, dict):
            raise ValueError(
                f"AWS secret {self._secret_name!r} must be a YAML/JSON "
                f"mapping, got {type(loaded).__name__}"
            )
        return loaded

    def _needs_fetch(self) -> bool:
        if self._values is None or self._fetched_at is None:
            return True
        if self._ttl_seconds <= 0:
            return False  # TTL disabled: cache for the process lifetime
        if (time.monotonic() - self._fetched_at) < self._ttl_seconds:
            return False
        # Outage backoff: serve stale without re-probing AWS on every read.
        if self._last_failed_fetch_at is not None and (
            time.monotonic() - self._last_failed_fetch_at
        ) < self._min_refresh_interval:
            return False
        return True

    def _current(self) -> dict:
        """Freshest available bundle. Only the first-ever load blocks: a reader that
        finds a re-fetch already in flight serves the stale bundle immediately."""
        if self._needs_fetch():
            first_load = self._values is None
            acquired = self._lock.acquire(blocking=first_load)
            if acquired:
                try:
                    if self._needs_fetch():
                        self._refetch_locked()
                finally:
                    self._lock.release()
        assert self._values is not None  # first fetch either populated it or raised
        return self._values

    def _refetch_locked(self) -> None:
        """One TTL re-fetch attempt. Caller holds ``self._lock``."""
        try:
            values = self._fetch()
        except ValueError:
            if self._values is None:
                raise
            self._last_failed_fetch_at = time.monotonic()
            # Config rot, not an outage — escalate so monitoring can tell them apart.
            logger.error(
                "AWS secret %r was rotated to invalid content; serving the cached "
                "bundle (age %.0fs) until the secret is fixed.",
                self._secret_name,
                time.monotonic() - (self._fetched_at or 0.0),
                exc_info=True,
            )
        except Exception:
            if self._values is None:
                raise
            self._last_failed_fetch_at = time.monotonic()
            logger.warning(
                "Re-fetching AWS secret %r failed; serving the cached bundle "
                "(age %.0fs). Will retry in %.0fs.",
                self._secret_name,
                time.monotonic() - (self._fetched_at or 0.0),
                self._min_refresh_interval,
                exc_info=True,
            )
        else:
            # Post-fetch clock read: the TTL measures from arrival, not request send.
            self._values = values
            self._fetched_at = time.monotonic()
            self._last_failed_fetch_at = None

    def _load(self) -> Mapping:
        """The bundle as a live view. Primes eagerly so a first-ever fetch failure
        raises at the call site, exactly as it did pre-TTL."""
        self._current()
        return self._view

    def refresh(self) -> Mapping:
        """Force a re-fetch (rate-limited to one per ``min_refresh_interval``) so a key
        added to AWS resolves on the first request that references it. Raises on
        failure — the caller asked for fresh data — leaving the cached bundle intact."""
        with self._lock:
            now = time.monotonic()
            within_cooldown = (
                self._last_forced_refresh_at is not None
                and (now - self._last_forced_refresh_at) < self._min_refresh_interval
            )
            if self._values is not None and within_cooldown:
                return self._view
            # Cooldown and TTL are stamped only after a successful fetch, off a post-fetch
            # clock read: a failed attempt must not suppress its own retry, and a fetch
            # slower than the cooldown must not hand back an already-expired one.
            try:
                values = self._fetch()
            except Exception:
                if self._values is not None:
                    self._last_failed_fetch_at = time.monotonic()  # arm the read-path backoff
                raise
            fetched_at = time.monotonic()
            self._values = values
            self._last_forced_refresh_at = fetched_at
            self._fetched_at = fetched_at
            self._last_failed_fetch_at = None
            return self._view

    def get(self, name: str) -> str | None:
        value = self._current().get(name)
        return None if value is None else str(value)
