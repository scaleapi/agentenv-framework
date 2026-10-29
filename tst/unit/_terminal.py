"""A person at a terminal for the install prompt, which the bundle and stored-task commands share."""

import os

from agent_env.cli import _install_prompt


def at_a_terminal(monkeypatch) -> dict:
    """Put stdin and stdout at a terminal, with an install that records what it's asked for and returns
    ``calls["code"]``, and a re-exec that records its argv and the rerun marker, then stops the command; the calls."""
    calls = {"added": [], "exec": [], "code": 0}

    def add(specs, **kwargs):
        calls["added"].append((specs, kwargs["yes"]))
        return calls["code"]

    def execv(path, argv):
        calls["exec"].append((path, argv, os.environ.get(_install_prompt._ASKED)))
        raise SystemExit(0)

    monkeypatch.setattr(_install_prompt, "_interactive", lambda: True)
    monkeypatch.setattr(_install_prompt._plugin_changes, "add", add)
    monkeypatch.setattr(_install_prompt.os, "execv", execv)
    monkeypatch.setenv(_install_prompt._ASKED, "")
    return calls
