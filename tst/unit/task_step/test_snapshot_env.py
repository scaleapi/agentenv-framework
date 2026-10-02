"""SnapshotEnvTaskStep: env resolution, retry-idempotent snapshot ids, and the
all-or-nothing export policy of the shared snapshot core."""
from __future__ import annotations

import json
import re

import pytest
from agentenv_protocol import FilePart
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.client import GetDataResponse

from agent_env.artifact import EnvironmentUniverseArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import set_object_store
from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, DeployedSandboxEnv
from agent_env.env.gateway.constants import EXT_STATE_URI, GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
from agent_env.store.object_store import S3ObjectStore
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps import snapshot_env as snapshot_mod
from agent_env.task_step.task_steps.snapshot_env import SnapshotEnvTaskStep
from tst.unit.store.fakes import FakeObjectStore


def _deployed(env_id="env-1", instance_id="env-1-abc123", gateway="https://gw-1"):
    return DeployedGatewayEnv(
        env_id=env_id,
        env_version=3,
        gateway_url=gateway,
        mcp_url=f"{gateway}/mcp",
        db_web_url=None,
        sandbox_id="sb-1",
        instance_id=instance_id,
        environment_card_url=f"{gateway}{WELL_KNOWN_PATH}",
        environment_card={"capabilities": {"extensions": GATEWAY_EXTENSIONS}},
    )


def _ctx(*envs, task_instance_id="task-x-65717eb66eeec083"):
    ctx = TaskStepContext()
    ctx.deployed_envs = list(envs)
    ctx.instance_id = task_instance_id
    return ctx


def _step(**overrides):
    fields = dict(id="snap-1", version=None)
    fields.update(overrides)
    return SnapshotEnvTaskStep(**fields)


# ── serialization + registry ─────────────────────────────────────────────────

def test_dict_roundtrip_and_registry():
    step = _step(
        env_id="env-1",
        env_instance_id="env-1-abc123",
        gateway_url="https://gw-override",
        snapshot_id="snap-explicit",
        original_universe_artifact_id="base-universe",
        export_timeout_seconds=120,
        include_env_trajectory=True,
    )
    data = step.to_dict()
    assert data["type"] == "snapshot_env"

    cls = get_task_step_registry()["snapshot_env"]
    restored = cls.from_dict(data)
    for attr in (
        "env_id", "env_instance_id", "gateway_url", "snapshot_id",
        "original_universe_artifact_id", "export_timeout_seconds",
        "include_env_trajectory",
    ):
        assert getattr(restored, attr) == getattr(step, attr)


def test_from_dict_defaults():
    step = SnapshotEnvTaskStep.from_dict({"id": "snap-1", "type": "snapshot_env"})
    assert step.env_id is None
    assert step.export_timeout_seconds == SnapshotEnvTaskStep.DEFAULT_EXPORT_TIMEOUT_SECONDS
    assert step.include_env_trajectory is False


# ── env resolution ───────────────────────────────────────────────────────────

def test_resolve_by_env_instance_id():
    target = _deployed(env_id="env-2", instance_id="env-2-xyz", gateway="https://gw-2")
    ctx = _ctx(_deployed(), target)
    assert _step(env_instance_id="env-2-xyz")._resolve_deployed_env(ctx) is target


def test_resolve_by_env_id():
    target = _deployed(env_id="env-2", instance_id="env-2-xyz", gateway="https://gw-2")
    ctx = _ctx(_deployed(), target)
    assert _step(env_id="env-2")._resolve_deployed_env(ctx) is target


def test_resolve_defaults_to_single_deployed_env():
    only = _deployed()
    assert _step()._resolve_deployed_env(_ctx(only)) is only


def test_resolve_ambiguous_without_selector_raises():
    ctx = _ctx(_deployed(), _deployed(env_id="env-2", instance_id="env-2-xyz"))
    with pytest.raises(RuntimeError, match="env_id, env_instance_id or env_step_id"):
        _step()._resolve_deployed_env(ctx)


def test_resolve_pure_override_mode_allows_missing_context_env():
    step = _step(env_id="env-9", gateway_url="https://gw-override")
    assert step._resolve_deployed_env(_ctx(_deployed())) is None


def test_resolve_unknown_instance_id_raises():
    with pytest.raises(RuntimeError, match="instance_id 'nope'"):
        _step(env_instance_id="nope")._resolve_deployed_env(_ctx(_deployed()))


# ── snapshot id derivation ───────────────────────────────────────────────────

def test_snapshot_id_derived_from_instance_id_is_deterministic():
    ctx = _ctx(_deployed())
    step = _step()
    first = step._derive_snapshot_id(ctx)
    second = step._derive_snapshot_id(ctx)
    assert first == second == "task-x-65717eb66eeec083__snapshot-snap-1"


def test_snapshot_id_override_wins():
    ctx = _ctx(_deployed())
    assert _step(snapshot_id="snap-mine")._derive_snapshot_id(ctx) == "snap-mine"


def test_snapshot_id_falls_back_to_an_adhoc_run_without_instance_id():
    ctx = _ctx(_deployed(), task_instance_id=None)
    first, second = _step()._derive_snapshot_id(ctx), _step()._derive_snapshot_id(ctx)
    assert re.fullmatch(r"adhoc-[0-9a-f]{12}__snapshot-snap-1", first)
    assert first != second


# ── execute wiring ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_refuses_an_env_deployed_without_a_gateway():
    bare = DeployedSandboxEnv(env_id="env-1", env_version=3, sandbox_id="srv", instance_id="env-1-abc123")
    with pytest.raises(RuntimeError, match="Deployed env 'env-1' has no gateway_url and no override was provided"):
        await _step(env_id="env-1").execute(_ctx(bare))

@pytest.mark.asyncio
async def test_execute_writes_summary_to_context(monkeypatch):
    captured = {}

    async def fake_snapshot_env_state(**kwargs):
        captured.update(kwargs)
        return snapshot_mod.EnvSnapshotResult(
            environment_universe_artifact_id=kwargs["snapshot_id"],
            environment_universe_artifact_version=1,
            environments_snapshotted=["slack", "gmail"],
            total=2,
        )

    class _FakeEnv:
        id = "env-1"
        version = 3

    monkeypatch.setattr(
        SnapshotEnvTaskStep, "snapshot_env_state", staticmethod(fake_snapshot_env_state)
    )
    monkeypatch.setattr(
        "agent_env.env.env.Env.get", classmethod(lambda cls, id, version=None: _FakeEnv())
    )

    ctx = _ctx(_deployed())
    step = _step(original_universe_artifact_id="base-universe")
    await step.execute(ctx)

    assert captured["gateway_url"] == "https://gw-1"
    assert captured["original_universe_artifact_id"] == "base-universe"
    assert ctx.metadata["env_snapshotted_universes"]["snap-1"] == {
        "id": "task-x-65717eb66eeec083__snapshot-snap-1",
        "version": 1,
    }


@pytest.mark.asyncio
async def test_execute_pins_deployed_env_version(monkeypatch):
    seen = {}

    async def fake_snapshot_env_state(**kwargs):
        return snapshot_mod.EnvSnapshotResult(
            environment_universe_artifact_id="s", environment_universe_artifact_version=1,
            environments_snapshotted=[], total=0,
        )

    class _FakeEnv:
        id = "env-1"
        version = 3

    def fake_get(cls, id, version=None):
        seen["env_id"], seen["version"] = id, version
        return _FakeEnv()

    monkeypatch.setattr(
        SnapshotEnvTaskStep, "snapshot_env_state", staticmethod(fake_snapshot_env_state)
    )
    monkeypatch.setattr("agent_env.env.env.Env.get", classmethod(fake_get))

    await _step().execute(_ctx(_deployed()))
    assert seen == {"env_id": "env-1", "version": 3}


# ── env trajectory capture ───────────────────────────────────────────────────

class _FakeTrajectoryResponse:
    def __init__(self, chunks, headers=None):
        self._chunks = chunks
        self.headers = headers or {}

    def raise_for_status(self):
        pass

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient; streams canned chunks or raises."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    def __call__(self, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, params=None):
        self.requests.append((method, url, params))
        client = self

        class _CM:
            async def __aenter__(_cm):
                if client.error:
                    raise client.error
                return client.response

            async def __aexit__(_cm, *exc):
                return False

        return _CM()


class _FakeObjectStore:
    def __init__(self):
        self.uploads = {}

    def put_file(self, key, file_path, content_type="application/octet-stream"):
        with open(file_path, "rb") as f:
            self.uploads[key] = (f.read(), content_type)
        return f"s3://fake-bucket/{key}"


class _FakeStoreConfig:
    def __init__(self, store, key_prefix=""):
        self._store = store
        self._key_prefix = key_prefix

    def get_object_store(self):
        return self._store

    def get_artifact_key_prefix(self):
        return self._key_prefix


def _wire_fake_snapshot(monkeypatch):
    async def fake_snapshot_env_state(**kwargs):
        return snapshot_mod.EnvSnapshotResult(
            environment_universe_artifact_id="s", environment_universe_artifact_version=1,
            environments_snapshotted=[], total=0,
        )

    class _FakeEnv:
        id = "env-1"
        version = 3

    monkeypatch.setattr(
        SnapshotEnvTaskStep, "snapshot_env_state", staticmethod(fake_snapshot_env_state)
    )
    monkeypatch.setattr(
        "agent_env.env.env.Env.get", classmethod(lambda cls, id, version=None: _FakeEnv())
    )


def _wire_fake_capture(monkeypatch, client, key_prefix=""):
    store = _FakeObjectStore()
    monkeypatch.setattr(snapshot_mod, "get_config", lambda: _FakeStoreConfig(store, key_prefix))
    monkeypatch.setattr(snapshot_mod.httpx, "AsyncClient", client)
    return store


@pytest.mark.asyncio
async def test_execute_skips_trajectory_capture_by_default(monkeypatch):
    _wire_fake_snapshot(monkeypatch)
    client = _FakeAsyncClient(response=_FakeTrajectoryResponse([b"{}\n"]))
    _wire_fake_capture(monkeypatch, client)

    ctx = _ctx(_deployed())
    await _step().execute(ctx)

    assert "env_trajectory" not in ctx.metadata
    assert client.requests == []


@pytest.mark.asyncio
async def test_trajectory_capture_persists_verbatim_and_summarizes(monkeypatch):
    """Verbatim JSONL in the store + an identifiers-only summary."""
    _wire_fake_snapshot(monkeypatch)
    line1 = b'{"event_type": "tool_call", "event_id": "event_1"}\n'
    line2 = b'{"event_id": "event_2", "virtual_time": "2026-01-02T00:00:00Z"}\n'
    payload = line1 + line2
    chunks = [payload[:30], payload[30:]]
    client = _FakeAsyncClient(response=_FakeTrajectoryResponse(chunks))
    store = _wire_fake_capture(monkeypatch, client)

    ctx = _ctx(_deployed())
    await _step(include_env_trajectory=True).execute(ctx)

    ((key, (data, content_type)),) = store.uploads.items()
    assert key.startswith("env_trajectory/instance_id=task-x-65717eb66eeec083/env-1-")
    assert key.endswith(".jsonl")
    assert data == payload
    assert content_type == "application/jsonl"

    entry = ctx.metadata["env_trajectory"]["env-1"]
    assert entry["object_url"] == f"s3://fake-bucket/{key}"
    assert entry["event_count"] == 2
    assert entry["capture_step_id"] == "snap-1"
    assert set(entry) == {"captured_at_utc", "capture_step_id", "object_url", "event_count"}
    (method, url, params) = client.requests[0]
    assert (method, url) == ("GET", "https://gw-1/trajectory")
    assert params is None


@pytest.mark.asyncio
async def test_trajectory_capture_is_kept_under_the_fixture_prefix(monkeypatch):
    _wire_fake_snapshot(monkeypatch)
    client = _FakeAsyncClient(response=_FakeTrajectoryResponse([b"{}\n"]))
    store = _wire_fake_capture(monkeypatch, client, key_prefix="fx/")

    await _step(include_env_trajectory=True).execute(_ctx(_deployed()))

    ((key, _),) = store.uploads.items()
    assert key.startswith("fx/env_trajectory/instance_id=task-x-65717eb66eeec083/env-1-")


@pytest.mark.asyncio
async def test_trajectory_capture_failure_is_fail_open(monkeypatch):
    """A fetch failure records {"error"} and the snapshot still succeeds."""
    _wire_fake_snapshot(monkeypatch)
    client = _FakeAsyncClient(error=RuntimeError("gateway unreachable"))
    store = _wire_fake_capture(monkeypatch, client)

    ctx = _ctx(_deployed())
    await _step(include_env_trajectory=True).execute(ctx)

    entry = ctx.metadata["env_trajectory"]["env-1"]
    assert entry["error"] == "RuntimeError: gateway unreachable"
    assert "object_url" not in entry
    assert store.uploads == {}
    # The fail-closed snapshot itself still ran to completion.
    assert ctx.metadata["env_snapshotted_universes"]["snap-1"]["version"] == 1


@pytest.mark.asyncio
async def test_trajectory_survives_a_failing_snapshot_export(monkeypatch):
    """Capture runs before the all-or-nothing export, so its metadata survives an export failure."""

    async def failing_snapshot_env_state(**kwargs):
        raise RuntimeError("1/2 services were not exported")

    class _FakeEnv:
        id = "env-1"
        version = 3

    monkeypatch.setattr(
        SnapshotEnvTaskStep, "snapshot_env_state", staticmethod(failing_snapshot_env_state)
    )
    monkeypatch.setattr(
        "agent_env.env.env.Env.get", classmethod(lambda cls, id, version=None: _FakeEnv())
    )
    client = _FakeAsyncClient(response=_FakeTrajectoryResponse([b"{}\n"]))
    store = _wire_fake_capture(monkeypatch, client)

    ctx = _ctx(_deployed())
    with pytest.raises(RuntimeError, match="not exported"):
        await _step(include_env_trajectory=True).execute(ctx)

    assert "object_url" in ctx.metadata["env_trajectory"]["env-1"]
    assert len(store.uploads) == 1


@pytest.mark.asyncio
async def test_trajectory_capture_records_one_sentence_when_the_card_offers_no_trajectory(monkeypatch):
    _wire_fake_snapshot(monkeypatch)
    client = _FakeAsyncClient(response=_FakeTrajectoryResponse([b"{}\n"]))
    store = _wire_fake_capture(monkeypatch, client)
    deployed = _deployed()
    deployed.environment_card = {"capabilities": {"extensions": [e for e in GATEWAY_EXTENSIONS if e["uri"] == EXT_STATE_URI]}}

    ctx = _ctx(deployed)
    await _step(include_env_trajectory=True).execute(ctx)

    assert ctx.metadata["env_trajectory"]["env-1"]["error"] == (
        "EnvCapabilityUnsupported: env 'env-1' does not offer 'get' on urn:agentenv:trajectory/v1.")
    assert client.requests == [] and store.uploads == {}


@pytest.mark.asyncio
async def test_trajectory_capture_streams_from_the_stored_cards_address_and_endpoint(monkeypatch):
    _wire_fake_snapshot(monkeypatch)
    client = _FakeAsyncClient(response=_FakeTrajectoryResponse([b"{}\n"]))
    _wire_fake_capture(monkeypatch, client)
    deployed = _deployed()
    deployed.environment_card_url = f"https://sandbox.example/sb-1{WELL_KNOWN_PATH}"
    deployed.environment_card = {"capabilities": {"extensions": [{
        "uri": "urn:agentenv:trajectory/v1", "params": {"endpoint": "/env/trajectory", "methods": {"get": {"method": "GET"}}}}]}}

    await _step(include_env_trajectory=True).execute(_ctx(deployed))

    assert [(m, u) for m, u, _ in client.requests] == [("GET", "https://sandbox.example/sb-1/env/trajectory")]


@pytest.mark.asyncio
@pytest.mark.parametrize("deployments", [[], [_deployed()]], ids=["pure-override", "next-to-a-deployment"])
async def test_trajectory_capture_with_a_gateway_url_override_reads_the_override(monkeypatch, deployments):
    _wire_fake_snapshot(monkeypatch)
    client = _FakeAsyncClient(response=_FakeTrajectoryResponse([b"{}\n"]))
    _wire_fake_capture(monkeypatch, client)

    await _step(env_id="env-1", gateway_url="https://override/", include_env_trajectory=True).execute(_ctx(*deployments))

    assert [(m, u) for m, u, _ in client.requests] == [("GET", "https://override/trajectory")]


# ── core: all-or-nothing + streaming guard ───────────────────────────────────

class _FakeServiceEnv:
    def __init__(self, name):
        self.environment_name = name


class _FakeMultiEnv:
    id = "env-1"
    version = 3

    def __init__(self, names):
        self.mcp_server_envs = [_FakeServiceEnv(n) for n in names]


@pytest.mark.asyncio
async def test_core_all_or_nothing_on_partial_failure(monkeypatch):
    async def fake_export(gateway_url, name, tmp_path, timeout, deployed=None):
        if name == "slack":
            raise ValueError("export-state returned non-JSON body (starts with b'<')")
        with open(tmp_path, "w") as f:
            json.dump({"ok": name}, f)
        return ".json"

    puts = []

    class _FakeFileArtifact:
        @staticmethod
        def put(id, *, description, file_path):
            puts.append(id)
            fa = type("FA", (), {})()
            fa.as_ref = lambda: None
            return fa

    class _FakeServiceArtifact:
        @staticmethod
        def put(id, *, environment_name, file_artifact):
            sa = type("SA", (), {})()
            sa.environment_name = environment_name
            return sa

    import agent_env.artifact as artifact_pkg

    monkeypatch.setattr(
        SnapshotEnvTaskStep, "_export_environment_to_file", staticmethod(fake_export)
    )
    monkeypatch.setattr(artifact_pkg, "FileArtifact", _FakeFileArtifact)
    monkeypatch.setattr(artifact_pkg, "EnvironmentArtifact", _FakeServiceArtifact)

    with pytest.raises(RuntimeError, match=r"1/2 services were not exported.*slack"):
        await SnapshotEnvTaskStep.snapshot_env_state(
            env=_FakeMultiEnv(["slack", "gmail"]),
            gateway_url="https://gw",
            snapshot_id="snap-x",
        )
    # The raise proves EnvironmentUniverseArtifact.put was never reached; per-service
    # artifacts created before the failure are harmless (versioned store).


@pytest.mark.asyncio
async def test_core_success_creates_universe_with_metadata(monkeypatch):
    async def fake_export(gateway_url, name, tmp_path, timeout, deployed=None):
        with open(tmp_path, "w") as f:
            json.dump({"ok": name}, f)
        return ".json"

    class _FakeFileArtifact:
        @staticmethod
        def put(id, *, description, file_path):
            return type("FA", (), {"as_ref": staticmethod(lambda: None)})()

    class _FakeServiceArtifact:
        @staticmethod
        def put(id, *, environment_name, file_artifact):
            sa = type("SA", (), {})()
            sa.environment_name = environment_name
            return sa

    universe_puts = {}
    carried_metadata = {"universe.yaml": object()}

    class _FakeUniverse:
        id = "snap-x"
        version = 4

        @staticmethod
        def put(id, *, environment_artifacts, metadata=None):
            universe_puts.update(
                id=id, names=[sa.environment_name for sa in environment_artifacts], metadata=metadata
            )
            return _FakeUniverse()

        @staticmethod
        def get(artifact_id, version=None):
            assert artifact_id == "base-universe"
            return type("Src", (), {"get_metadata": staticmethod(lambda: carried_metadata)})()

    import agent_env.artifact as artifact_pkg

    monkeypatch.setattr(
        SnapshotEnvTaskStep, "_export_environment_to_file", staticmethod(fake_export)
    )
    monkeypatch.setattr(artifact_pkg, "FileArtifact", _FakeFileArtifact)
    monkeypatch.setattr(artifact_pkg, "EnvironmentArtifact", _FakeServiceArtifact)
    monkeypatch.setattr(artifact_pkg, "EnvironmentUniverseArtifact", _FakeUniverse)

    result = await SnapshotEnvTaskStep.snapshot_env_state(
        original_universe_artifact_id="base-universe",
        env=_FakeMultiEnv(["slack", "gmail"]),
        gateway_url="https://gw",
        snapshot_id="snap-x",
    )

    assert universe_puts["id"] == "snap-x"
    assert sorted(universe_puts["names"]) == ["gmail", "slack"]
    assert universe_puts["metadata"] is carried_metadata
    assert result.environment_universe_artifact_id == "snap-x"
    assert result.total == 2


@pytest.mark.asyncio
async def test_core_zip_export_creates_zip_filename_artifact(monkeypatch):
    """A FilePart(zip) export flows through as a .zip-suffixed FileArtifact path,
    so FileArtifact.filename (= basename) ends in .zip — the load side routes a
    .zip file:// artifact to reset_data() for a lossless round-trip."""
    async def fake_export(gateway_url, name, tmp_path, timeout, deployed=None):
        # Simulate the lossless-zip case: write zip bytes and report ".zip".
        with open(tmp_path, "wb") as f:
            f.write(b"PK\x03\x04zip-bytes")
        return ".zip"

    put_paths = {}

    class _FakeFileArtifact:
        @staticmethod
        def put(id, *, description, file_path):
            put_paths[id] = file_path
            return type("FA", (), {"as_ref": staticmethod(lambda: None)})()

    class _FakeServiceArtifact:
        @staticmethod
        def put(id, *, environment_name, file_artifact):
            sa = type("SA", (), {})()
            sa.environment_name = environment_name
            return sa

    class _FakeUniverse:
        id = "snap-z"
        version = 1

        @staticmethod
        def put(id, *, environment_artifacts, metadata=None):
            return _FakeUniverse()

    import agent_env.artifact as artifact_pkg

    monkeypatch.setattr(
        SnapshotEnvTaskStep, "_export_environment_to_file", staticmethod(fake_export)
    )
    monkeypatch.setattr(artifact_pkg, "FileArtifact", _FakeFileArtifact)
    monkeypatch.setattr(artifact_pkg, "EnvironmentArtifact", _FakeServiceArtifact)
    monkeypatch.setattr(artifact_pkg, "EnvironmentUniverseArtifact", _FakeUniverse)

    result = await SnapshotEnvTaskStep.snapshot_env_state(
        env=_FakeMultiEnv(["slack"]),
        gateway_url="https://gw",
        snapshot_id="snap-z",
    )

    assert result.total == 1
    # FileArtifact.filename = os.path.basename(file_path); must end in .zip.
    assert put_paths["snap-z-slack-file"].endswith(".zip")


@pytest.mark.asyncio
async def test_export_service_to_file_materializes_filepart_zip(monkeypatch, tmp_path):
    """_export_service_to_file consumes a v1 FilePart (base64 .zip bytes),
    writes the raw zip bytes to disk, and returns the ".zip" suffix."""
    import base64

    from agentenv_protocol import FilePart
    from agentenv_protocol.client import GetDataResponse
    from agent_env.task_step.task_steps import snapshot_env as mod

    zip_bytes = b"PK\x03\x04lossless-zip-bundle"
    part = FilePart(
        file={
            "bytes": base64.b64encode(zip_bytes).decode(),
            "name": "slack.zip",
            "mimeType": "application/zip",
        }
    )

    async def fake_supports_v1(base_url):
        return True

    async def fake_get_data(base_url, timeout=None):
        return GetDataResponse(parts=[part])

    async def fake_get_card(base_url, timeout=10):
        return {}  # no s3-credentials extension -> _push_s3_credentials no-ops

    from agentenv_protocol import client as protocol_v1

    monkeypatch.setattr(protocol_v1, "supports_v1", fake_supports_v1)
    monkeypatch.setattr(protocol_v1, "get_card", fake_get_card)
    monkeypatch.setattr(protocol_v1, "get_data", fake_get_data)

    out = str(tmp_path / "export-tmp")
    suffix = await mod.SnapshotEnvTaskStep._export_environment_to_file(
        "https://gw", "slack", out, timeout_seconds=30
    )

    assert suffix == ".zip"
    with open(out, "rb") as f:
        assert f.read() == zip_bytes


@pytest.mark.asyncio
async def test_export_service_to_file_resolves_relative_uri_and_streams(
    monkeypatch, tmp_path
):
    """A FilePart with a RELATIVE uri (e.g. "export-snapshot") is resolved against
    the service base_url and streamed to disk — so multi-GB bundles never ride
    inline in the JSON-RPC response."""
    from agentenv_protocol import FilePart
    from agentenv_protocol.client import GetDataResponse
    from agentenv_protocol import client as protocol_v1
    from agent_env.env import legacy_protocol
    from agent_env.task_step.task_steps import snapshot_env as mod

    zip_bytes = b"PK\x03\x04streamed-from-export-snapshot"
    part = FilePart(
        file={"uri": "export-snapshot", "name": "gdrive.zip", "mimeType": "application/zip"}
    )

    async def fake_supports_v1(base_url):
        return True

    async def fake_get_data(base_url, timeout=None):
        return GetDataResponse(parts=[part])

    async def fake_get_card(base_url, timeout=10):
        return {}  # no s3-credentials extension -> _push_s3_credentials no-ops

    monkeypatch.setattr(protocol_v1, "supports_v1", fake_supports_v1)
    monkeypatch.setattr(protocol_v1, "get_card", fake_get_card)
    monkeypatch.setattr(protocol_v1, "get_data", fake_get_data)

    captured = {}

    class _FakeResp:
        def raise_for_status(self):
            pass

        async def aiter_bytes(self):
            yield zip_bytes

    class _FakeStreamCtx:
        async def __aenter__(self):
            return _FakeResp()

        async def __aexit__(self, *exc):
            return False

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url, timeout=None):
            captured["method"] = method
            captured["url"] = url
            return _FakeStreamCtx()

    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda *a, **k: _FakeClient())

    out = str(tmp_path / "export-tmp")
    suffix = await mod.SnapshotEnvTaskStep._export_environment_to_file(
        "https://gw", "gdrive", out, timeout_seconds=30
    )

    assert suffix == ".zip"
    with open(out, "rb") as f:
        assert f.read() == zip_bytes
    expected_base = legacy_protocol.environment_base_url("https://gw", "gdrive", mcp=True)
    assert captured["method"] == "GET"
    assert captured["url"] == f"{expected_base.rstrip('/')}/export-snapshot"


def _serve_filepart_uri(monkeypatch, uri):
    part = FilePart(file={"uri": uri, "name": "gdrive.zip", "mimeType": "application/zip"})

    async def fake_supports_v1(base_url):
        return True

    async def fake_get_data(base_url, timeout=None):
        return GetDataResponse(parts=[part])

    async def fake_get_card(base_url, timeout=10):
        return {}

    def no_http(*a, **k):
        raise AssertionError("this uri must not be streamed")

    monkeypatch.setattr(protocol_v1, "supports_v1", fake_supports_v1)
    monkeypatch.setattr(protocol_v1, "get_card", fake_get_card)
    monkeypatch.setattr(protocol_v1, "get_data", fake_get_data)
    monkeypatch.setattr(snapshot_mod.httpx, "AsyncClient", no_http)


@pytest.mark.asyncio
async def test_export_service_to_file_returns_an_object_url_as_the_services_upload(monkeypatch, tmp_path):
    """A FilePart uri with any scheme but http(s) is the service's own upload: handed back as-is,
    nothing streamed or written."""
    store = FakeObjectStore()
    set_object_store(store)
    uploaded = store.put("snapshots/gdrive.zip", b"PK\x03\x04")
    _serve_filepart_uri(monkeypatch, uploaded)

    out = tmp_path / "export-tmp"
    result = await SnapshotEnvTaskStep._export_environment_to_file("https://gw", "gdrive", str(out), 30)

    assert result == uploaded
    assert not out.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("upload_url", ["fake://home/snapshots/gdrive#v2.zip", "fake://shared/snapshots/gdrive#v2.zip"])
async def test_core_registers_a_service_upload_in_place_and_it_loads_back(local_stores, monkeypatch, tmp_path, upload_url):
    """Registered where the service put it, in the store's own root or anywhere else the store
    can read (no re-upload), named by the url's last segment so a '#' survives (urlparse would
    end the path there), and readable through the store."""
    store = FakeObjectStore()
    set_object_store(store)
    reset_artifact_store()
    bundle_file = tmp_path / "bundle.zip"
    bundle_file.write_bytes(b"PK\x03\x04bundle")
    uploaded = store.put_file_at(upload_url, str(bundle_file))

    async def fake_export(gateway_url, name, tmp_path, timeout, deployed=None):
        return uploaded

    monkeypatch.setattr(SnapshotEnvTaskStep, "_export_environment_to_file", staticmethod(fake_export))

    result = await SnapshotEnvTaskStep.snapshot_env_state(
        env=_FakeMultiEnv(["gdrive"]), gateway_url="https://gw", snapshot_id="snap-g",
    )

    (bundle,) = EnvironmentUniverseArtifact.get(result.environment_universe_artifact_id).get_file_artifacts().values()
    assert bundle.object_url == uploaded
    assert bundle.filename == "gdrive#v2.zip"
    assert bundle.content_type == "application/zip"
    assert bundle.load() == b"PK\x03\x04bundle"
    assert list(store.objects) == [uploaded]


@pytest.mark.asyncio
async def test_core_fails_a_service_whose_upload_the_store_cannot_find(local_stores, monkeypatch):
    set_object_store(FakeObjectStore())
    reset_artifact_store()

    async def fake_export(gateway_url, name, tmp_path, timeout, deployed=None):
        return "fake://shared/snapshots/missing.zip"

    monkeypatch.setattr(SnapshotEnvTaskStep, "_export_environment_to_file", staticmethod(fake_export))

    with pytest.raises(RuntimeError, match="failed to register the uploaded bundle"):
        await SnapshotEnvTaskStep.snapshot_env_state(
            env=_FakeMultiEnv(["gdrive"]), gateway_url="https://gw", snapshot_id="snap-g",
        )


@pytest.mark.asyncio
async def test_core_snapshots_under_an_id_longer_than_a_filename_can_be(local_stores, monkeypatch):
    """A run's snapshot id can pass the 255-byte filename limit, so neither a temp file nor a key segment of the
    local store may be the whole id."""
    snapshot_id = f"task-{'x' * 250}__snapshot-snap-1"

    async def fake_export(gateway_url, name, tmp_path, timeout, deployed=None):
        with open(tmp_path, "w") as f:
            json.dump({"service": name}, f)
        return ".json"

    monkeypatch.setattr(SnapshotEnvTaskStep, "_export_environment_to_file", staticmethod(fake_export))

    result = await SnapshotEnvTaskStep.snapshot_env_state(env=_FakeMultiEnv(["slack"]), gateway_url="https://gw", snapshot_id=snapshot_id)

    (service,) = EnvironmentUniverseArtifact.get(result.environment_universe_artifact_id).get_environment_artifacts()
    assert (result.environment_universe_artifact_id, service.id) == (snapshot_id, f"{snapshot_id}-slack")
    assert json.loads(service.get_file_artifact().load()) == {"service": "slack"}


@pytest.mark.asyncio
async def test_export_service_to_file_filepart_without_bytes_or_uri_raises_valueerror(
    monkeypatch, tmp_path
):
    """A FilePart with neither bytes nor uri raises ValueError (which export_one
    catches to fail just this service), not TypeError from stream("GET", None)."""
    from agentenv_protocol import FilePart
    from agentenv_protocol.client import GetDataResponse
    from agent_env.task_step.task_steps import snapshot_env as mod
    from agentenv_protocol import client as protocol_v1

    # The protocol model requires bytes XOR uri, so a "neither" part can't be
    # built directly — null the uri post-construction to simulate the edge.
    part = FilePart(file={"uri": "http://x", "name": "slack.zip", "mimeType": "application/zip"})
    part.file.uri = None

    async def fake_supports_v1(base_url):
        return True

    async def fake_get_data(base_url, timeout=None):
        return GetDataResponse(parts=[part])

    async def fake_get_card(base_url, timeout=10):
        return {}  # no s3-credentials extension -> _push_s3_credentials no-ops

    monkeypatch.setattr(protocol_v1, "supports_v1", fake_supports_v1)
    monkeypatch.setattr(protocol_v1, "get_card", fake_get_card)
    monkeypatch.setattr(protocol_v1, "get_data", fake_get_data)

    with pytest.raises(ValueError, match="neither bytes nor uri"):
        await mod.SnapshotEnvTaskStep._export_environment_to_file(
            "https://gw", "slack", str(tmp_path / "export-tmp"), timeout_seconds=30
        )


@pytest.mark.asyncio
async def test_export_reads_the_child_env_from_the_stored_card(monkeypatch, tmp_path):
    """A child env on the stored card: S3 creds and data/get go to the card's address, with no card read."""
    from agent_env.task_step.task_steps import snapshot_env as mod

    s3_ext = {"uri": "urn:agentenv:add-s3-credentials/v1",
              "params": {"endpoint": "/agentenv/ext/add_s3_credentials", "methods": {"add_s3_credentials": {"method": "POST"}}}}
    record = _carded(await _card({"mcp-gdrive": [s3_ext]}))
    sent = _mock_http(monkeypatch, rpc_result={"parts": [{"kind": "file", "file": {"uri": "export-snapshot", "name": "gdrive.zip"}}]},
                      body=b"PK\x03\x04card-bundle")
    _fake_aws(monkeypatch)

    out = str(tmp_path / "export-tmp")
    suffix = await mod.SnapshotEnvTaskStep._export_environment_to_file("https://gw-1", "gdrive", out, 30, record)

    assert suffix == ".zip"
    assert [(r.method, str(r.url)) for r in sent] == [
        ("POST", f"{_CARD}/svc/mcp-gdrive/agentenv/ext/add_s3_credentials"),
        ("POST", f"{_CARD}/svc/mcp-gdrive/agentenv"),
        ("GET", f"{_CARD}/svc/mcp-gdrive/export-snapshot"),
    ]
    assert json.loads(sent[0].content) == {"aws_access_key_id": "AK", "aws_secret_access_key": "SK", "aws_session_token": None,
                                           "region_name": None, "bucket": "bucket"}
    assert json.loads(sent[1].content)["method"] == "data/get"
    with open(out, "rb") as f:
        assert f.read() == b"PK\x03\x04card-bundle"


@pytest.mark.asyncio
async def test_export_of_a_child_env_missing_from_the_card_streams_legacy_export_state(monkeypatch, tmp_path):
    from agent_env.task_step.task_steps import snapshot_env as mod

    # gdrive serves no card, so the gateway leaves it out of the composed card.
    record = _carded(await _card({"mcp-slack": []}))
    sent = _mock_http(monkeypatch, body=b'{"files": []}')

    out = str(tmp_path / "export-tmp")
    suffix = await mod.SnapshotEnvTaskStep._export_environment_to_file("https://gw-1", "gdrive", out, 30, record)

    assert suffix == ".json"
    assert [(r.method, str(r.url)) for r in sent] == [("GET", "https://gw-1/svc/mcp-gdrive/export-state")]


@pytest.mark.asyncio
async def test_snapshot_uses_the_stored_card_only_for_the_deployments_own_gateway(monkeypatch):
    seen = []

    async def fake_export(gateway_url, name, tmp_path, timeout, deployed=None):
        seen.append((gateway_url, deployed))
        raise ValueError("stop after the export call")

    monkeypatch.setattr(SnapshotEnvTaskStep, "_export_environment_to_file", staticmethod(fake_export))
    record = _carded({"name": "gw", "children_environments": []})
    record.gateway_url = "https://gw-1/"
    for gateway_url in ("https://gw-1/", "https://elsewhere"):
        with pytest.raises(RuntimeError, match="not exported"):
            await SnapshotEnvTaskStep.snapshot_env_state(
                env=_FakeMultiEnv(["slack"]), gateway_url=gateway_url, snapshot_id="snap-x", deployed=record,
            )

    assert seen == [("https://gw-1", record), ("https://elsewhere", None)]


@pytest.mark.asyncio
async def test_export_with_a_record_that_has_no_stored_card_keeps_the_live_probe(monkeypatch, tmp_path):
    import dataclasses

    from agent_env.task_step.task_steps import snapshot_env as mod

    record = dataclasses.replace(_carded({}), environment_card=None)
    sent = _mock_http(monkeypatch, rpc_result={"parts": [{"kind": "data", "data": {"rows": 1}}]}, body=b'{"name": "gdrive", "url": "/agentenv"}')

    suffix = await mod.SnapshotEnvTaskStep._export_environment_to_file("https://gw-1", "gdrive", str(tmp_path / "out"), 30, record)

    assert suffix == ".json"
    assert [(r.method, str(r.url)) for r in sent] == [
        ("GET", "https://gw-1/svc/mcp-gdrive/.well-known/agent-env.json"),
        ("GET", "https://gw-1/svc/mcp-gdrive/.well-known/agent-env.json"),
        ("POST", "https://gw-1/svc/mcp-gdrive/agentenv"),
    ]


def test_enumerate_services_single_mcp_server_env():
    env = _FakeServiceEnv("calendar")
    env.id, env.version = "cal-env", 1
    assert SnapshotEnvTaskStep._enumerate_environments(env) == ["calendar"]


def test_enumerate_services_no_services_raises():
    env = type("E", (), {"id": "x", "version": 1})()
    with pytest.raises(RuntimeError, match="no MCP services"):
        SnapshotEnvTaskStep._enumerate_environments(env)


# ── loaded-universe metadata carry-over ──────────────────────────────────────

def test_resolve_loaded_universe_ref_reads_store(monkeypatch):
    import agent_env.env.store as store_mod

    class _FakeStore:
        def get_environment_universe(self, instance_id):
            assert instance_id == "env-1-abc123"
            return {"id": "u1", "version": 7}

    monkeypatch.setattr(store_mod, "get_env_instance_store", lambda: _FakeStore())
    ref = SnapshotEnvTaskStep._resolve_loaded_universe_ref(_deployed(instance_id="env-1-abc123"))
    assert ref == ("u1", 7)


def test_resolve_loaded_universe_ref_none_without_instance_id():
    assert SnapshotEnvTaskStep._resolve_loaded_universe_ref(None) is None
    assert SnapshotEnvTaskStep._resolve_loaded_universe_ref(_deployed(instance_id=None)) is None


def test_resolve_loaded_universe_ref_swallows_store_error(monkeypatch):
    """Store failure degrades to None (best-effort), never crashes the snapshot."""
    import agent_env.env.store as store_mod

    class _FakeStore:
        def get_environment_universe(self, instance_id):
            raise RuntimeError("mongo down")

    monkeypatch.setattr(store_mod, "get_env_instance_store", lambda: _FakeStore())
    assert SnapshotEnvTaskStep._resolve_loaded_universe_ref(_deployed()) is None


def test_resolve_loaded_universe_ref_none_when_instance_never_loaded(monkeypatch):
    import agent_env.env.store as store_mod

    class _FakeStore:
        def get_environment_universe(self, instance_id):
            return None  # instance exists but never recorded a universe

    monkeypatch.setattr(store_mod, "get_env_instance_store", lambda: _FakeStore())
    assert SnapshotEnvTaskStep._resolve_loaded_universe_ref(_deployed()) is None


def _install_core_artifact_fakes(monkeypatch, universe_cls):
    """Wire the export + FileArtifact/EnvironmentArtifact fakes shared by the core
    carry-over tests; the caller supplies the EnvironmentUniverseArtifact fake."""
    async def fake_export(gateway_url, name, tmp_path, timeout, deployed=None):
        with open(tmp_path, "w") as f:
            json.dump({"ok": name}, f)
        return ".json"

    class _FakeFileArtifact:
        @staticmethod
        def put(id, *, description, file_path):
            return type("FA", (), {"as_ref": staticmethod(lambda: None)})()

    class _FakeServiceArtifact:
        @staticmethod
        def put(id, *, environment_name, file_artifact):
            sa = type("SA", (), {})()
            sa.environment_name = environment_name
            return sa

    import agent_env.artifact as artifact_pkg

    monkeypatch.setattr(
        SnapshotEnvTaskStep, "_export_environment_to_file", staticmethod(fake_export)
    )
    monkeypatch.setattr(artifact_pkg, "FileArtifact", _FakeFileArtifact)
    monkeypatch.setattr(artifact_pkg, "EnvironmentArtifact", _FakeServiceArtifact)
    monkeypatch.setattr(artifact_pkg, "EnvironmentUniverseArtifact", universe_cls)


@pytest.mark.asyncio
async def test_core_carries_metadata_from_loaded_universe(monkeypatch):
    """With no explicit original id, the snapshot carries metadata (version-pinned)
    from the universe the env instance recorded loading."""
    import agent_env.env.store as store_mod

    carried = {"universe.yaml": object(), "metadata.json": object()}
    got = {}

    class _FakeUniverse:
        id = "snap-x"
        version = 4

        @staticmethod
        def put(id, *, environment_artifacts, metadata=None):
            got["metadata"] = metadata
            return _FakeUniverse()

        @staticmethod
        def get(artifact_id, version=None):
            got["get"] = (artifact_id, version)
            return type("Src", (), {"get_metadata": staticmethod(lambda: carried)})()

    class _FakeStore:
        def get_environment_universe(self, instance_id):
            got["instance_id"] = instance_id
            return {"id": "loaded-universe", "version": 2}

    _install_core_artifact_fakes(monkeypatch, _FakeUniverse)
    monkeypatch.setattr(store_mod, "get_env_instance_store", lambda: _FakeStore())

    await SnapshotEnvTaskStep.snapshot_env_state(
        env=_FakeMultiEnv(["slack"]),
        gateway_url="https://gw",
        snapshot_id="snap-x",
        deployed=_deployed(instance_id="env-1-abc123"),
    )

    assert got["instance_id"] == "env-1-abc123"
    assert got["get"] == ("loaded-universe", 2)  # version-pinned to what was loaded
    assert got["metadata"] is carried


@pytest.mark.asyncio
async def test_core_explicit_id_wins_over_loaded_universe(monkeypatch):
    """An explicit original_universe_artifact_id overrides the instance-store one."""
    import agent_env.env.store as store_mod

    got = {}

    class _FakeUniverse:
        id = "snap-x"
        version = 4

        @staticmethod
        def put(id, *, environment_artifacts, metadata=None):
            got["metadata"] = metadata
            return _FakeUniverse()

        @staticmethod
        def get(artifact_id, version=None):
            got["get"] = (artifact_id, version)
            return type("Src", (), {"get_metadata": staticmethod(lambda: {"universe.yaml": 1})})()

    class _FakeStore:
        def get_environment_universe(self, instance_id):
            raise AssertionError("instance store must not be consulted when id is explicit")

    _install_core_artifact_fakes(monkeypatch, _FakeUniverse)
    monkeypatch.setattr(store_mod, "get_env_instance_store", lambda: _FakeStore())

    await SnapshotEnvTaskStep.snapshot_env_state(
        original_universe_artifact_id="explicit-base",
        env=_FakeMultiEnv(["slack"]),
        gateway_url="https://gw",
        snapshot_id="snap-x",
        deployed=_deployed(),
    )
    assert got["get"] == ("explicit-base", None)  # explicit id, unpinned


@pytest.mark.asyncio
async def test_core_metadata_best_effort_when_instance_has_no_universe(monkeypatch):
    """No explicit id and no recorded universe → snapshot still succeeds, metadata=None."""
    import agent_env.env.store as store_mod

    got = {}

    class _FakeUniverse:
        id = "snap-x"
        version = 4

        @staticmethod
        def put(id, *, environment_artifacts, metadata=None):
            got["metadata"] = metadata
            return _FakeUniverse()

        @staticmethod
        def get(artifact_id, version=None):
            raise AssertionError("get must not be called when no source id resolved")

    class _FakeStore:
        def get_environment_universe(self, instance_id):
            return None

    _install_core_artifact_fakes(monkeypatch, _FakeUniverse)
    monkeypatch.setattr(store_mod, "get_env_instance_store", lambda: _FakeStore())

    result = await SnapshotEnvTaskStep.snapshot_env_state(
        env=_FakeMultiEnv(["slack"]),
        gateway_url="https://gw",
        snapshot_id="snap-x",
        deployed=_deployed(),
    )
    assert got["metadata"] is None
    assert result.total == 1


@pytest.mark.asyncio
async def test_core_explicit_bad_original_id_fails_fast(monkeypatch):
    """A bad explicit id surfaces (fails before exports); the auto path never would."""
    class _FakeUniverse:
        @staticmethod
        def put(id, *, environment_artifacts, metadata=None):
            raise AssertionError("put must not be reached")

        @staticmethod
        def get(artifact_id, version=None):
            raise RuntimeError("no such universe")

    _install_core_artifact_fakes(monkeypatch, _FakeUniverse)

    with pytest.raises(RuntimeError, match="no such universe"):
        await SnapshotEnvTaskStep.snapshot_env_state(
            original_universe_artifact_id="bad",
            env=_FakeMultiEnv(["slack"]),
            gateway_url="https://gw",
            snapshot_id="snap-x",
            deployed=_deployed(),
        )


# The card's address differs from gateway_url, so an export that reads the card is told apart from one that builds /svc/... itself.
_CARD = "https://sandbox.example/sandbox/sb-1-18765"


async def _card(children: dict[str, list]) -> dict:
    """The real gateway's composed card, with a v1 backing server behind each key and the given extensions."""
    from agent_env.env.gateway.gateway import Gateway

    gw = Gateway(host="127.0.0.1", port=0, server_name="env1234", internal_mcp_servers=[],
                 rest_proxy_urls={key: f"http://{key}:18765" for key in children})

    async def fetch(client, key, base_url):
        card = {"name": key.removeprefix("mcp-"), "url": "/agentenv", "capabilities": {"extensions": children[key]}}
        return gw._rewrite_child_card(key, card)

    gw._fetch_child_card = fetch
    return json.loads((await gw._serve_env_card()).body)


def _carded(card: dict) -> DeployedEnv:
    from agent_env.env.gateway.constants import WELL_KNOWN_PATH

    return DeployedGatewayEnv(
        env_id="env-1", env_version=3, gateway_url="https://gw-1", mcp_url="https://gw-1/mcp", db_web_url=None,
        sandbox_id="sb-1", environment_card_url=f"{_CARD}{WELL_KNOWN_PATH}", environment_card=card,
    )


def _mock_http(monkeypatch, *, rpc_result: dict | None = None, body: bytes = b"{}"):
    """Record every request: JSON-RPC calls get `rpc_result`, extension calls an empty object, anything else `body`."""
    import httpx

    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path.endswith("/agentenv"):
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": rpc_result or {"parts": []}})
        if "/ext/" in request.url.path:
            return httpx.Response(200, json={})
        return httpx.Response(200, content=body)

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent


class _SharingS3Store(S3ObjectStore):
    def shared_credentials_env(self) -> dict[str, str]:
        return {"AWS_ACCESS_KEY_ID": "AK", "AWS_SECRET_ACCESS_KEY": "SK"}


def _fake_aws(monkeypatch):
    """An S3 object store that shares AWS creds, so the S3-credentials push actually sends."""
    store = _SharingS3Store(client=object(), bucket="bucket")
    monkeypatch.setattr(snapshot_mod, "get_config", lambda: type("Cfg", (), {"get_object_store": lambda self: store})())
