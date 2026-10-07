"""Keeping the model key out of a Sailbox: Sail adds it to the agent's model requests as they leave.

The key is stored as a Sail secret named from its SHA-256, so one key is one secret and different keys never
share one. The secret is never deleted, so no launch can remove one another, concurrent or in another
process, still needs; a Sailbox's saved policy, which only names it, is deleted with the Sailbox. A saved egress policy on the Sailbox sets it as the auth headers of requests to the model endpoint's
host. Inside the Sailbox the key is replaced by a placeholder, and containers start trusting the CA Sail
terminates those requests' TLS with.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
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
# A model-key env var assignment and the whole shell word assigned: quoted, escaped and bare parts alike.
_KEY_ASSIGNMENT = re.compile(
    r"\b(?:LITELLM_API_KEY|ANTHROPIC_API_KEY)=((?:'[^']*'|\"(?:[^\"\\]|\\.)*\"|\\.|[^\s'\"\\])*)"
)
# A whole word that is one variable reference a shell expands: $NAME, ${NAME}, "$NAME" or "${NAME}".
_REFERENCE = re.compile(r'^"?\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})"?$')

# By secret name: the key, so a handle reconnected here can still scrub its agent's key, and the launches
# and Sailboxes in this process that need it, so the key is forgotten once none does. Process memory only.
_injected_keys: dict[str, str] = {}
_holders: dict[str, set[str]] = {}


def secret_name(key: str) -> str:
    return SECRET_PREFIX + hashlib.sha256(key.encode()).hexdigest()[:32].upper()


def carries_model_key(text: str, *, shell: bool) -> bool:
    """Whether ``text`` assigns a model-key env var a value other than the placeholder. In a ``shell`` script
    the assigned word is decoded as the shell would (quotes and escapes), and a word that is wholly one
    unquoted or double-quoted variable reference is not a value; in any other text, every value is literal."""
    for word in _KEY_ASSIGNMENT.findall(text):
        if shell and _REFERENCE.match(word):
            continue
        if shell:
            try:
                value = "".join(shlex.split(word))
            except ValueError:
                return True
        else:
            value = word
        if value and value != PLACEHOLDER:
            return True
    return False


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
                "use an https endpoint or set inject_model_key = false in [sandbox.providers.sail_vm.config]"
            )
        return cls(host=parsed.hostname, secret=secret_name(key), key=key)

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

    def hold(self, holder: str) -> None:
        """Record ``holder`` (a launch or a Sailbox id) as needing this injection's secret."""
        if self.key is not None:
            _injected_keys[self.secret] = self.key
        _holders.setdefault(self.secret, set()).add(holder)

    def release(self, holder: str) -> None:
        """Drop ``holder``, forgetting the key once no launch or Sailbox in this process needs it."""
        holders = _holders.get(self.secret, set())
        holders.discard(holder)
        if not holders:
            _holders.pop(self.secret, None)
            _injected_keys.pop(self.secret, None)

    def recover_key(self, candidate: str | None) -> None:
        """Recover the key this injection's secret was named from: one this process injected, else
        ``candidate`` when it is that key."""
        self.key = _injected_keys.get(self.secret)
        if self.key is None and candidate and secret_name(candidate) == self.secret:
            self.key = candidate

    def _encodings(self) -> list[str]:
        """The key as agent-env writes it into commands: shell-quoted, inside single quotes, and raw."""
        return [shlex.quote(self.key), self.key.replace("'", "'\\''"), self.key]

    def scrub(self, text: str) -> str:
        for encoded in self._encodings() if self.key else ():
            text = text.replace(encoded, PLACEHOLDER)
        return text

    def scrub_bytes(self, data: bytes) -> bytes:
        for encoded in self._encodings() if self.key else ():
            data = data.replace(encoded.encode(), PLACEHOLDER.encode())
        return data
