"""Shared stubs for the agent-state capture tests.

Follows the ``tst.unit.store.fakes`` convention: the capture path touches an
artifact store, an object store (including its grants) and the agent sidecar, and
both ``test_agent_state_capture`` and ``test_prompt_agent_periodic_snapshot`` need
the same fakes for all three.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Optional

import httpx
from agentenv_protocol.transfers import HttpPutGrant

from agent_env.a2a_agent import A2AAgent
from agent_env.env.env import DeployedGatewayEnv
from agent_env.task_step.context import TaskStepContext

BUCKET = "artifact-bucket"


def agent_card(
    *,
    snapshot: bool | dict = True,
    snapshot_endpoint: Optional[str] = None,
    trajectory: Optional[dict] = None,
) -> dict:
    """An AgentCard advertising the snapshot and/or trajectory extensions.

    ``trajectory`` is the ``get`` method entry verbatim, so a test can advertise
    the ``context_id`` mode, a ``task_id``-only one, or a bare ``{}``.
    """
    extensions: list[dict] = []
    if snapshot:
        params: dict[str, Any] = dict(snapshot) if isinstance(snapshot, dict) else {}
        if snapshot_endpoint is not None:
            params["endpoint"] = snapshot_endpoint
        extensions.append({"uri": A2AAgent.EXT_SNAPSHOT, "params": params})
    if trajectory is not None:
        extensions.append(
            {"uri": A2AAgent.EXT_TRAJECTORY, "params": {"methods": {"get": trajectory}}}
        )
    return {"capabilities": {"extensions": extensions}}


# A `get` that advertises the cumulative-across-the-session mode the periodic
# read needs, in the `oneOf` shape a real card uses.
CONTEXT_ID_GET = {"request": {"oneOf": [{"required": ["task_id"]}, {"required": ["context_id"]}]}}


class _StubUniverse:
    def __init__(self, id: str, version: int, s3_url: str):
        self.id = id
        self.version = version
        self.bundle_object_url = s3_url


class _StubObjectStore:
    supports_transfer_grants = True
    max_single_upload_bytes = None

    def __init__(self):
        self.puts: list[tuple[str, bytes]] = []
        self.write_grants: list[dict] = []
        self.withheld: set[str] = set()

    def list_at(self, prefix: str) -> list[str]:
        """What the agent uploaded: every object it was granted, less the ``withheld`` names."""
        return [
            grant["object_url"]
            for grant in self.write_grants
            if grant["object_url"].startswith(prefix)
            and grant["object_url"].rsplit("/", 1)[-1] not in self.withheld
        ]

    def object_url(self, key: str) -> str:
        return f"s3://{BUCKET}/{key}"

    def grants_reach(self, sandbox_type: str | None) -> bool:
        return True

    def issue_write_grant(self, object_url, *, media_type, max_bytes, expires_in):
        self.write_grants.append(
            {
                "object_url": object_url,
                "media_type": media_type,
                "max_bytes": max_bytes,
                "expires_in": expires_in,
            }
        )
        return HttpPutGrant(
            kind="http-put",
            url="https://objects.example.test/write?secret=signed",
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )

    def get_object_key(self, url: str) -> str:
        prefix = f"s3://{BUCKET}/"
        if not url.startswith(prefix):
            raise ValueError(f"{url} is not in the configured bucket {BUCKET}")
        return url[len(prefix):]

    def put(self, key: str, body: bytes, content_type: str | None = None) -> str:
        self.puts.append((key, body))
        return f"s3://{BUCKET}/{key}"


class CaptureRecorder:
    """What the stubs saw, plus the knobs a test turns to force a failure."""

    def __init__(self):
        self.save_requests: list[dict] = []
        self.trajectory_requests: list[dict] = []
        self.universes: list[str] = []
        self.object_store = _StubObjectStore()
        self.next_version_calls: list[str] = []
        self._version = 2
        self.key_prefix = ""
        # Overridable per test.
        self.save_status = 200
        self.save_body: Optional[dict] = None
        self.trajectory_status = 200
        self.trajectory_body: dict = {"trajectory": [{"role": "user"}]}

    def next_version(self, artifact_id: str) -> int:
        self.next_version_calls.append(artifact_id)
        self._version += 1
        return self._version


def install_capture_stubs(monkeypatch) -> CaptureRecorder:
    """Stub the artifact store, object store and the sidecar HTTP."""
    import agent_env.artifact.artifacts.file_artifact_universe as fau_mod
    import agent_env.artifact.store as artifact_store_mod
    import agent_env.config as config_mod

    rec = CaptureRecorder()

    class _Cfg:
        def get_object_store(self):
            return rec.object_store

        def get_artifact_key_prefix(self):
            return rec.key_prefix

    monkeypatch.setattr(config_mod, "get_config", lambda: _Cfg())
    monkeypatch.setattr(
        "agent_env.task_step.snapshot_utils.agent_state_capture.get_config",
        lambda: _Cfg(),
    )
    monkeypatch.setattr(artifact_store_mod, "get_artifact_store", lambda: rec)

    class _FAU:
        @staticmethod
        def put_existing(*, id: str, s3_url: str):
            # Mirrors the real one: version is minted by the store, and the
            # bundle url is the prefix that was actually written.
            rec.universes.append(s3_url)
            return _StubUniverse(id=id, version=rec._version, s3_url=s3_url)

    monkeypatch.setattr(fau_mod, "FileArtifactUniverse", _FAU)

    # Patched at `request`, not `post`, so a caller reaching for either is covered
    # — httpx's `post` routes through `request`.
    async def fake_request(self, method, url, *, json=None, timeout=None, **kwargs):  # noqa: A002
        request = httpx.Request(method, url)
        if url.endswith("/ext/trajectory"):
            rec.trajectory_requests.append(
                {"method": method, "url": url, "json": json, "timeout": timeout}
            )
            return httpx.Response(
                rec.trajectory_status, json=rec.trajectory_body, request=request
            )
        rec.save_requests.append(
            {"method": method, "url": url, "json": json, "timeout": timeout}
        )
        body = rec.save_body
        if body is None:
            request_json = json or {}
            body = {
                "context_id": request_json.get("context_id"),
                "objects": {name: {"size_bytes": 1} for name in request_json.get("objects", {})},
            }
        return httpx.Response(rec.save_status, json=body, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)
    return rec


class _StubAgent:
    def __init__(self, name="solver", a2a_url="https://agent", card=None):
        self.agent_name = name
        self.a2a_url = a2a_url
        self.api_url = a2a_url
        self.a2a_card = card if card is not None else agent_card()
        self.sandbox_id = "sb-agent"
        self.sandbox_type = None


def context(*, agent: Optional[_StubAgent] = None, env_id: Optional[str] = None) -> TaskStepContext:
    ctx = TaskStepContext()
    ctx.deployed_agents = [agent or _StubAgent()]
    if env_id:
        ctx.deployed_envs = [
            DeployedGatewayEnv(
                env_id=env_id,
                env_version=1,
                gateway_url="https://gw",
                mcp_url="https://gw/mcp",
                db_web_url=None,
                sandbox_id="sb-env",
            )
        ]
    return ctx
