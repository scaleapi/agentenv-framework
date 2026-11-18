from __future__ import annotations

import subprocess

import pytest

from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandbox


def recorded_sandbox_ids(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    made, init = [], LocalSandbox.__init__

    def recorded(self, *args, **kwargs):
        init(self, *args, **kwargs)
        made.append(self.sandbox_id)

    monkeypatch.setattr(LocalSandbox, "__init__", recorded)
    return made


def container_names_carrying(sandbox_ids: list[str]) -> set[str]:
    names = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True,
                           check=True).stdout
    return {name for name in names.split() if any(sandbox_id in name for sandbox_id in sandbox_ids)}
