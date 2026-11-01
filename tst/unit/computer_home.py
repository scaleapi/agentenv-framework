"""A computer at its driver's boundary, for steps that reach inside one: its home a folder on this machine, ``data/add``
writing there, ``data/get`` reading there, and computer-exec/v1 ``run`` running commands there; each call recorded,
and the address each ``data/add`` is sent to.
``pkill`` matches nothing, so one test never ends another's processes on this machine."""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
from pathlib import Path

from agentenv_protocol import DATA_OBJECTS_EXTENSION_URI, FilePart
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer import COMPUTER_EXEC
from agentenv_protocol.types import FileWithBytes, GetDataResponse

from agent_env.env.env import DeployedSandboxEnv

PORT = 18765
CARD = {"name": "desk", "capabilities": {"operations": ["data/reset", "data/add", "data/get"], "extensions": [
    {"uri": COMPUTER_EXEC, "params": {"endpoint": "/computer/exec", "methods": {"run": {}},
                                      "max_output_bytes": 1 << 20, "max_timeout_s": 3600}}]}}


class Computer:
    def __init__(self, home: Path):
        self.home = home
        self.calls: list[tuple] = []
        self.added_at: list[str] = []
        self.add_fails: Exception | None = None

    def deployed(self, env_id: str = "desk", card: dict | None = None, **computer) -> DeployedSandboxEnv:
        return DeployedSandboxEnv(env_id=env_id, env_version=1, env_provider_type="computer",
                                  environment_card_url="http://127.0.0.1:20001/.well-known/agent-env.json",
                                  environment_card=card or CARD, sandbox_id="local-1", sandbox_type="local",
                                  metadata={"computer": {"port": PORT, **computer}})

    def install(self, monkeypatch) -> Computer:
        monkeypatch.setattr(protocol_v1, "add_data", self.add_data)
        monkeypatch.setattr(protocol_v1, "get_data", self.get_data)
        monkeypatch.setattr(protocol_v1, "invoke_extension", self.invoke_extension)
        return self

    async def add_data(self, base_url, parts, timeout=120, verify=True):
        self.calls.append(("data/add", [p.metadata[DATA_OBJECTS_EXTENSION_URI] for p in parts], timeout))
        self.added_at.append(base_url)
        if self.add_fails is not None:
            raise self.add_fails
        for part in parts:
            if isinstance(part.file, FileWithBytes):
                target = self.home / part.metadata[DATA_OBJECTS_EXTENSION_URI]["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(part.file.bytes))

    async def get_data(self, base_url, timeout=30, verify=True, *, write_namespace=None, paths=None):
        self.calls.append(("data/get", paths))
        return GetDataResponse(parts=[
            FilePart(file=FileWithBytes(bytes=base64.b64encode((self.home / path).read_bytes()).decode()),
                     metadata={DATA_OBJECTS_EXTENSION_URI: {"path": path}})
            for path in paths if (self.home / path).is_file()])

    async def invoke_extension(self, base_url, card, uri, params=None, timeout=30, verify=True, *, method=None):
        assert (uri, method) == (COMPUTER_EXEC, "run")
        self.calls.append(("run", params, timeout))
        return await asyncio.to_thread(self._run, params)

    def _run(self, params: dict) -> dict:
        if params["argv"][0] == "pkill":
            return {"exit_code": 1, "stdout": "", "stderr": "", "truncated": False, "timed_out": False}
        try:
            done = subprocess.run(params["argv"], cwd=self.home, env={**os.environ, **(params.get("env") or {})},
                                  capture_output=True, text=True, timeout=params["timeout_s"])
        except subprocess.TimeoutExpired:
            return {"exit_code": None, "stdout": "", "stderr": "", "truncated": False, "timed_out": True}
        return {"exit_code": done.returncode, "stdout": done.stdout, "stderr": done.stderr, "truncated": False,
                "timed_out": False}
