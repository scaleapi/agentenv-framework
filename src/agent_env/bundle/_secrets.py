"""Secrets a bundle import leaves out: env vars that name or hold one, and credentials anywhere else in a value."""

from __future__ import annotations

import re
from typing import Any

from agent_env.config.describe import is_secret_key

# The fields that map environment variable names to values, where a credential is most often set.
_ENV_VAR_MAPS = ("env_vars", "default_env_vars")
# Each credential a value can hold, how to find it and what masks it: a URL's user info when it holds a password or is
# long enough to be a token rather than a user name such as git, then values in the shape their issuers give them.
_CREDENTIALS = (
    ("a URL's credentials", re.compile(r"(?P<scheme>[a-zA-Z][\w+.-]*://)(?:[^/@\s]*:[^/@\s]*|[^/@\s:]{16,})@"),
     r"\g<scheme>***@"),
    ("an API key", re.compile(r"\bsk[-_](?=[\w-]*\d)[\w-]{20,}"), "***"),
    ("a GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_\w{22,})"), "***"),
    ("an AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "***"),
    ("a Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"), "***"),
    ("a Google API key", re.compile(r"\bAIza[\w-]{35}"), "***"),
    ("a Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "***"),
    ("a JWT", re.compile(r"\beyJ[\w-]{8,}\.eyJ[\w-]{8,}\.[\w-]{8,}"), "***"),
    ("a private key", re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"
                                 r"(?:.*?-----END (?:[A-Z]+ )*PRIVATE KEY-----|.*)", re.DOTALL), "***"),
)


def scrub(value: Any, path: str = "") -> list[tuple[str, str]]:
    """Drop secret env vars from ``value`` and mask credentials elsewhere in it, in place; each one's path and why."""
    removed = []
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, item in list(items):
        at = f"{path}.{key}" if path and isinstance(key, str) else f"{path}[{key}]" if path else str(key)
        if isinstance(value, dict) and key in _ENV_VAR_MAPS and isinstance(item, dict):
            for name, setting in list(item.items()):
                secret = is_secret_key(name)
                found = _masked(setting)[1] if isinstance(setting, str) and not secret else []
                if secret or found:
                    del item[name]
                    why = "its name looks like a secret" if secret else f"its value holds {found[0]}"
                    removed.append((f"{at}.{name}", f"dropped: {why}"))
        elif isinstance(item, str):
            masked, found = _masked(item)
            if found:
                value[key] = masked
                removed.append((at, f"masked: {', '.join(found)}"))
        else:
            removed += scrub(item, at)
    return removed


def _masked(text: str) -> tuple[str, list[str]]:
    """``text`` with each credential replaced by ``***``, and the kinds it held."""
    found = []
    for what, pattern, mask in _CREDENTIALS:
        text, count = pattern.subn(mask, text)
        if count:
            found.append(what)
    return text, found
