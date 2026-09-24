"""Shared stubs for the agent-state capture tests.

Follows the ``tst.unit.store.fakes`` convention: the capture path touches an
artifact store, an object store (including its presigner) and the agent sidecar, and
both ``test_agent_state_capture`` and ``test_prompt_agent_periodic_snapshot`` need
the same fakes for all three.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx

from agent_env.a2a_agent import A2AAgent
from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext

BUCKET = "artifact-bucket"


def agent_card(
    *,
    snapshot: bool = True,
    snapshot_endpoint: Optional[str] = None,
    trajectory: Optional[dict] = None,
) -> dict:
    """An AgentCard advertising the snapshot and/or trajectory extensions.

    ``trajectory`` is the ``get`` method entry verbatim, so a test can advertise
    the ``context_id`` mode, a ``task_id``-only one, or a bare ``{}``.
    """
    extensions: list[dict] = []
    if snapshot:
        params: dict[str, Any] = {}
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
    def __init__(self):
        self.puts: list[tuple[str, bytes]] = []
        self.signed_posts: list[dict] = []

    def object_url(self, key: str) -> str:
        return f"s3://{BUCKET}/{key}"

    def signed_post(self, url_prefix, *, expires_in=3600, max_bytes=None):
        self.signed_posts.append(
            {"url_prefix": url_prefix, "expires_in": expires_in, "max_bytes": max_bytes}
        )
        key = f"{self.get_object_key(url_prefix)}${{filename}}"
        return {"url": f"https://{BUCKET}.s3.amazonaws.com/", "fields": {"key": key}}

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
        self.object_store = _StubObjectStore()
        self.next_version_calls: list[str] = []
        self._version = 2
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
        def get_s3_bucket(self):
            return BUCKET

        def get_object_store(self):
            return rec.object_store

    monkeypatch.setattr(config_mod, "get_config", lambda: _Cfg())
    monkeypatch.setattr(artifact_store_mod, "get_artifact_store", lambda: rec)

    class _FAU:
        @staticmethod
        def put_existing(*, id: str, s3_url: str):
            # Mirrors the real one: version is minted by the store, and the
            # bundle url is the prefix that was actually written.
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
        # Callable form: echo a prefix relative to the issued one, which is random.
        if callable(body):
            body = body(json or {})
        if body is None:
            body = {"s3_prefix": (json or {}).get("s3_prefix")}
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


def context(*, agent: Optional[_StubAgent] = None, env_id: Optional[str] = None) -> TaskStepContext:
    ctx = TaskStepContext()
    ctx.deployed_agents = [agent or _StubAgent()]
    if env_id:
        ctx.deployed_envs = [
            DeployedEnv(
                env_id=env_id,
                env_version=1,
                gateway_url="https://gw",
                mcp_url="https://gw/mcp",
                db_web_url=None,
                sandbox_id="sb-env",
            )
        ]
    return ctx

