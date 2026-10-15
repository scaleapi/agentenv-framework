"""Keeping the model key, and the endpoint's ``[model.headers]``, out of a Sailbox: Sail adds them to the
agent's model requests as they leave.

The key is stored as a Sail secret named from its SHA-256, so one key is one secret and different keys never
share one; each header value is a secret of its own, named the same way from its name and value. The secret is never deleted, so no launch can remove one another, concurrent or in another
process, still needs; a Sailbox's saved policy, which only names it, is deleted with the Sailbox. A saved egress policy on the Sailbox sets it as the auth headers of requests to the model endpoint's
host. Inside the Sailbox the key is replaced by a placeholder, and containers start trusting the CA Sail
terminates those requests' TLS with.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse

from agent_env.config.model import MODEL_HEADERS_ENV

SECRET_PREFIX = "AGENTENV_LITELLM_"
#: Hex characters of a value's SHA-256 a secret name keeps.
SECRET_HASH_CHARS = 32
HEADER_SECRET_PREFIX = f"{SECRET_PREFIX}H_"
POLICY_PREFIX = "agentenv-"
PLACEHOLDER = "sail-injected-model-key"
HEADER_PLACEHOLDER = "sail-injected-header"
KEY_ENV = "LITELLM_API_KEY"
BASE_URL_ENV = "LITELLM_BASE_URL"
#: The headers the key is set as; a [model.headers] entry may not set them too.
KEY_HEADERS = ("authorization", "x-api-key")

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
_KEY_NAMES = ("LITELLM_API_KEY", "ANTHROPIC_API_KEY")
# A shell word: quoted, escaped and bare parts, concatenated.
_SHELL_WORD = re.compile(r"(?:'[^']*'|\"(?:[^\"\\]|\\.)*\"|\\.|[^\s'\"\\;|&<>()])+")
# An assigned value, as written, that is wholly one variable reference a shell expands: $NAME, ${NAME}, "$NAME", "${NAME}".
_REFERENCE = re.compile(r'^"?\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})"?$')
# In text that isn't a shell script: KEY=value, or a YAML / JSON KEY: value.
_LITERAL_ASSIGNMENT = re.compile(
    r"\b(?:LITELLM_API_KEY|ANTHROPIC_API_KEY)['\"]?(?:=|\s*:\s*)['\"]?([^\s'\",}#]*)"
)
#: Shorter keys are refused: scrubbing replaces every occurrence, which for a short one hits unrelated text.
MIN_KEY_LENGTH = 16

# By secret name: the key, so a handle reconnected here can still scrub its agent's key, and the launches
# and Sailboxes in this process that need it, so the key is forgotten once none does. Process memory only.
_injected_keys: dict[str, str] = {}
_holders: dict[str, set[str]] = {}
# By holder: the headers it was launched or reconnected with, under their secrets, so a handle reconnected here
# can scrub them; each holder drops its own, so Sailboxes sharing headers never drop one another's.
_held_headers: dict[str, tuple[frozenset[str], dict[str, str], frozenset[str]]] = {}
# In text that isn't a shell script: the headers' JSON, to the end of its line.
_LITERAL_HEADERS = re.compile(rf"\b{MODEL_HEADERS_ENV}['\"]?(?:=|\s*:\s*)(.*)")


def secret_name(key: str) -> str:
    return SECRET_PREFIX + hashlib.sha256(key.encode()).hexdigest()[:SECRET_HASH_CHARS].upper()


def header_secret_name(name: str, value: str) -> str:
    digest = hashlib.sha256(f"{name.lower()}\0{value}".encode()).hexdigest()
    return HEADER_SECRET_PREFIX + digest[:SECRET_HASH_CHARS].upper()


def _is_header_placeholder(value: str) -> bool:
    """Whether the headers' JSON ``value`` carries no header value but the placeholder."""
    try:
        headers = json.loads(value)
    except ValueError:
        return False
    return isinstance(headers, dict) and all(v == HEADER_PLACEHOLDER for v in headers.values())


def carries_model_key(text: str, *, shell: bool) -> bool:
    """Whether ``text`` assigns a model-key env var a value other than the placeholder. In a ``shell`` script
    each word is decoded as the shell would (quotes and escapes, in the name too), and a value that is wholly
    one unquoted or double-quoted variable reference isn't one; in other text, every value is literal."""
    if not shell:
        return (any(value and value != PLACEHOLDER for value in _LITERAL_ASSIGNMENT.findall(text))
                or _carries_headers_literally(text))
    for word in _SHELL_WORD.findall(text):
        try:
            decoded = "".join(shlex.split(word))
        except ValueError:
            return True
        name, assigned, value = decoded.partition("=")
        if not assigned or not value or _REFERENCE.match(_written_value(word)):
            continue
        if name in _KEY_NAMES and value != PLACEHOLDER:
            return True
        if name == MODEL_HEADERS_ENV and not _is_header_placeholder(value):
            return True
    return False


def _carries_headers_literally(text: str) -> bool:
    """Whether text that isn't a shell script (a file, one argv word) assigns the headers' env var a value."""
    for value in _LITERAL_HEADERS.findall(text):
        value = value.strip().strip("'\"")
        if value and not _REFERENCE.match(value) and not _is_header_placeholder(value):
            return True
    return False


def _written_value(word: str) -> str:
    """The part of a shell word after its first ``=`` outside single quotes and escapes, as written."""
    quote = None
    index = 0
    while index < len(word):
        char = word[index]
        if quote == "'":
            if char == "'":
                quote = None
        elif char == "=":
            return word[index + 1:]
        elif char == "\\":
            index += 1
        elif char == '"':
            quote = None if quote == '"' else '"'
        elif char == "'":
            quote = "'"
        index += 1
    return ""


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
    """The model endpoint host, the Sail secrets holding the key and the endpoint's headers, and the saved
    policy applying them. ``key`` is None on a reconnected Sailbox unless the configured key is the one
    injected, and ``header_values`` unless the configured headers are."""

    host: str
    secret: str
    key: str | None = field(default=None, repr=False)
    #: Each header's lowercased name -> the Sail secret holding its value.
    headers: dict[str, str] = field(default_factory=dict)
    #: The headers as agent-env hands them to workloads, names as configured: what scrubbing replaces.
    header_values: dict[str, str] | None = field(default=None, repr=False)
    #: ``LITELLM_EXTRA_HEADERS`` exactly as each deploy wrote it, where that isn't the JSON agent-env writes.
    header_texts: frozenset[str] = field(default=frozenset(), repr=False)
    #: The model endpoint's path as a prefix (``/v1/``) the headers are scoped to, as they are in agent-env;
    #: None when it has no path. The key goes with every request to the host either way.
    path_prefix: str | None = None
    policy_id: str | None = None
    #: Saved policies a swap may or may not have applied (Sail couldn't say), deleted once it can.
    unsettled_policy_ids: list[str] = field(default_factory=list)
    #: Every saved policy made for one Sailbox is named under this prefix, so teardown from any handle, in
    #: any process, finds them all; a reconnected handle takes it from the applied policy's name.
    policy_prefix: str = field(default_factory=lambda: f"{POLICY_PREFIX}{uuid.uuid4().hex[:12]}")

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
        if len(key) < MIN_KEY_LENGTH:
            raise ValueError(
                f"Sail model-key injection needs a {KEY_ENV} of at least {MIN_KEY_LENGTH} characters, to scrub it "
                "safely; set inject_model_key = false in [sandbox.providers.sail_vm.config] to pass it in"
            )
        text = env.get(MODEL_HEADERS_ENV) or ""
        try:
            values = json.loads(text or "{}")
        except ValueError:
            values = None
        if not isinstance(values, dict) or not all(isinstance(value, str) for value in values.values()):
            raise ValueError(f"{MODEL_HEADERS_ENV} must be a JSON object of header names and values")
        if clash := sorted(name for name in values if name.lower() in KEY_HEADERS):
            raise ValueError(
                f"[model.headers] sets {clash}, which Sail sets from the model key; drop them, or set "
                "inject_model_key = false in [sandbox.providers.sail_vm.config] to pass headers and key in"
            )
        path = parsed.path.rstrip("/")
        return cls(host=parsed.hostname, secret=secret_name(key), key=key,
                   headers={name.lower(): header_secret_name(name, value) for name, value in values.items()},
                   header_values=values or None,
                   header_texts=frozenset({text}) if values and text != json.dumps(values) else frozenset(),
                   path_prefix=f"{path}/" if values and path else None)

    @classmethod
    def from_document(cls, document: Any, policy_id: str | None) -> ModelKeyInjection | None:
        """The injection a saved policy document applies, or None when its rules aren't one."""
        rules = document.get("rules") if isinstance(document, dict) else None
        if not isinstance(rules, dict) or len(rules) != 1:
            return None
        (host, entries), = rules.items()
        if not isinstance(entries, list) or len(entries) not in (1, 2):
            return None
        try:
            refs = dict(entries[0]["request"]["set"]["headers"])
            prefix = entries[0]["match"]["path"]["prefix"] if len(entries) == 2 else None
        except (TypeError, ValueError, KeyError, AttributeError):
            return None
        refs.pop("authorization", None)  # "Bearer ${secrets.…}", the key again: rules() below checks it
        names = {header: _SECRET_REF.fullmatch(ref) if isinstance(ref, str) else None for header, ref in refs.items()}
        if "x-api-key" not in names or not all(names.values()):
            return None
        secret = names.pop("x-api-key").group(1)
        headers = {header: match.group(1) for header, match in names.items()}
        injection = cls(host=host, secret=secret, headers=headers, path_prefix=prefix, policy_id=policy_id)
        own = secret.startswith(SECRET_PREFIX) and all(s.startswith(HEADER_SECRET_PREFIX) for s in headers.values())
        return injection if own and rules == injection.rules() else None

    def rules(self) -> dict[str, Any]:
        """Sail egress rules setting the key as both auth headers model endpoints read on every request to the
        host, and each header on those under the endpoint's path."""
        ref = f"${{secrets.{self.secret}}}"
        key = {"request": {"set": {"headers": {"authorization": f"Bearer {ref}", "x-api-key": ref}}}}
        if not self.headers:
            return {self.host: [key]}
        both = {"request": {"set": {"headers": {
            **key["request"]["set"]["headers"], **{name: f"${{secrets.{secret}}}" for name, secret in self.headers.items()},
        }}}}
        if self.path_prefix is None:
            return {self.host: [both]}
        return {self.host: [{"match": {"path": {"prefix": self.path_prefix}}, **both}, key]}

    def header_secrets(self) -> dict[str, str]:
        """Each header's Sail secret -> its value, for the launch to store."""
        return {header_secret_name(name, value): value for name, value in (self.header_values or {}).items()}

    def matches(self, other: ModelKeyInjection | None) -> bool:
        return other is not None and (other.host, other.secret, other.headers, other.path_prefix) == (
            self.host, self.secret, self.headers, self.path_prefix)

    def _headers_id(self) -> frozenset[str]:
        return frozenset(self.headers.values())

    def hold(self, holder: str) -> None:
        """Record ``holder`` (a launch or a Sailbox id) as needing this injection's key, and its headers."""
        if self.key is not None:
            _injected_keys[self.secret] = self.key
        _holders.setdefault(self.secret, set()).add(holder)
        if self.header_values:
            _held_headers[holder] = (self._headers_id(), self.header_values, self.header_texts)

    def release(self, holder: str) -> None:
        """Drop ``holder``, forgetting the key once no launch or Sailbox in this process needs it, and the
        headers ``holder`` held."""
        holders = _holders.get(self.secret, set())
        holders.discard(holder)
        if not holders:
            _holders.pop(self.secret, None)
            _injected_keys.pop(self.secret, None)
        _held_headers.pop(holder, None)

    def recover_key(self, candidate: str | None) -> None:
        """Recover the key this injection's secret was named from: one this process injected, else
        ``candidate`` when it is that key."""
        self.key = _injected_keys.get(self.secret)
        if self.key is None and candidate and secret_name(candidate) == self.secret:
            self.key = candidate

    def recover_headers(self, candidate: Mapping[str, str] | None) -> None:
        """Recover the headers this injection's secrets were named from: ones this process injected, else
        ``candidate`` (the configured headers) when they are those."""
        if not self.headers:
            return
        held = [(values, texts) for headers, values, texts in _held_headers.values() if headers == self._headers_id()]
        if held:
            self.header_values = held[0][0]
            self.header_texts = frozenset().union(*(texts for _, texts in held))
        elif candidate and {
            name.lower(): header_secret_name(name, value) for name, value in candidate.items()
        } == self.headers:
            self.header_values = dict(candidate)

    def _encodings(self) -> list[tuple[str, str]]:
        """Each value agent-env writes into commands, and the placeholder it becomes, in every encoding:
        shell-quoted, inside single quotes, and raw. The headers go whole, as their JSON in each spelling, so
        a short value can't rewrite unrelated text, and first, so a header holding the key is still whole."""
        pairs = []
        if self.header_values:
            placeholders = json.dumps(dict.fromkeys(self.header_values, HEADER_PLACEHOLDER))
            pairs += [(text, placeholders) for text in [json.dumps(self.header_values), *sorted(self.header_texts)]]
        if self.key:
            pairs.append((self.key, PLACEHOLDER))
        return [(encode(value), encode(placeholder)) for value, placeholder in pairs
                for encode in (shlex.quote, lambda text: text.replace("'", "'\\''"), lambda text: text)]

    def scrub(self, text: str) -> str:
        for encoded, placeholder in self._encodings():
            text = text.replace(encoded, placeholder)
        return text

    def scrub_bytes(self, data: bytes) -> bytes:
        for encoded, placeholder in self._encodings():
            data = data.replace(encoded.encode(), placeholder.encode())
        return data
