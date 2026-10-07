"""Keeping the model key out of a Sailbox: Sail adds it to the agent's model requests as they leave.

The key is stored as a Sail secret named from its SHA-256, so one key is one secret and different keys never
share one. A saved egress policy on the Sailbox sets it as the auth headers of requests to the model endpoint's
host. Inside the Sailbox the key is replaced by a placeholder, and containers start trusting the CA Sail
terminates those requests' TLS with.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse

SECRET_PREFIX = "AGENTENV_LITELLM_"
PLACEHOLDER = "sail-injected-model-key"
KEY_ENV = "LITELLM_API_KEY"
BASE_URL_ENV = "LITELLM_BASE_URL"

#: The Sailbox's CA bundle, which includes the CA Sail intercepts TLS with, and where containers see it.
VM_CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"
CONTAINER_CA_BUNDLE = "/etc/ssl/certs/sailbox-ca-bundle.crt"
CONTAINER_TRUST_ENV = {
    "SSL_CERT_FILE": CONTAINER_CA_BUNDLE,
    "REQUESTS_CA_BUNDLE": CONTAINER_CA_BUNDLE,
    "NODE_EXTRA_CA_CERTS": CONTAINER_CA_BUNDLE,
    "CURL_CA_BUNDLE": CONTAINER_CA_BUNDLE,
}
#: Ahead of /usr/bin on the Sailbox's PATH, so every ``docker run`` / ``create`` passes through it.
DOCKER_SHIM_PATH = "/usr/local/bin/docker"

_SECRET_REF = re.compile(r"\$\{secrets\.([A-Za-z0-9_]+)\}")

# Keys this process has injected, by secret name, so a handle reconnected here can still scrub its own
# agent's key. Process memory only: the keys are already held here, and nothing is persisted.
_injected_keys: dict[str, str] = {}


def secret_name(key: str) -> str:
    return SECRET_PREFIX + hashlib.sha256(key.encode()).hexdigest()[:32].upper()


def docker_shim() -> str:
    """A ``docker`` wrapper that gives every container it runs or creates the Sailbox's CA bundle."""
    trust = " ".join(f"-e {name}={value}" for name, value in CONTAINER_TRUST_ENV.items())
    flags = f"-v {VM_CA_BUNDLE}:{CONTAINER_CA_BUNDLE}:ro {trust}"
    return f"""#!/bin/sh
case "$1 $2" in
  "container run"*|"container create"*) sub="$1 $2"; shift 2; exec /usr/bin/docker $sub {flags} "$@" ;;
esac
case "$1" in
  run|create) sub="$1"; shift; exec /usr/bin/docker "$sub" {flags} "$@" ;;
esac
exec /usr/bin/docker "$@"
"""


@dataclass
class ModelKeyInjection:
    """The model endpoint host, the Sail secret holding the key, and the saved policy applying it.
    ``key`` is None on a reconnected Sailbox unless the configured key is the one injected."""

    host: str
    secret: str
    key: str | None = field(default=None, repr=False)
    policy_id: str | None = None

    @classmethod
    def for_env(cls, env: Mapping[str, str]) -> ModelKeyInjection | None:
        """The injection for an agent ``env`` carrying a model key, or None when it carries none."""
        key = env.get(KEY_ENV)
        if not key or key == PLACEHOLDER:
            return None
        base_url = env.get(BASE_URL_ENV) or ""
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError(
                f"Sail injects the model key only into HTTPS requests, but {BASE_URL_ENV} is {base_url!r}; "
                "use an https endpoint or set inject_model_key = false in [sandbox.providers.sail.config]"
            )
        injection = cls(host=parsed.hostname, secret=secret_name(key), key=key)
        _injected_keys[injection.secret] = key
        return injection

    @classmethod
    def from_document(cls, document: Any, policy_id: str | None) -> ModelKeyInjection | None:
        """The injection a saved policy document applies, or None when its rules aren't one."""
        rules = document.get("rules") if isinstance(document, dict) else None
        if not isinstance(rules, dict) or len(rules) != 1:
            return None
        (host,) = rules
        names = set(_SECRET_REF.findall(json.dumps(rules)))
        if len(names) != 1:
            return None
        (name,) = names
        injection = cls(host=host, secret=name, policy_id=policy_id)
        return injection if name.startswith(SECRET_PREFIX) and rules == injection.rules() else None

    def rules(self) -> dict[str, Any]:
        """Sail egress rules setting the key as both auth headers model endpoints read."""
        ref = f"${{secrets.{self.secret}}}"
        return {self.host: [{"request": {"set": {"headers": {"authorization": f"Bearer {ref}", "x-api-key": ref}}}}]}

    def matches(self, other: ModelKeyInjection | None) -> bool:
        return other is not None and (other.host, other.secret) == (self.host, self.secret)

    def recover_key(self, candidate: str | None) -> None:
        """Recover the key this injection's secret was named from: one this process injected, else
        ``candidate`` when it is that key."""
        self.key = _injected_keys.get(self.secret)
        if self.key is None and candidate and secret_name(candidate) == self.secret:
            self.key = candidate

    def scrub(self, text: str) -> str:
        return text.replace(self.key, PLACEHOLDER) if self.key else text

    def scrub_bytes(self, data: bytes) -> bytes:
        return data.replace(self.key.encode(), PLACEHOLDER.encode()) if self.key else data
