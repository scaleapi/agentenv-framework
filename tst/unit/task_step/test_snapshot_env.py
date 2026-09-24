"""SnapshotEnvTaskStep: env resolution, retry-idempotent snapshot ids, and the
all-or-nothing export policy of the shared snapshot core."""
from __future__ import annotations

import json

import pytest

from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps import snapshot_env as snapshot_mod
from agent_env.task_step.task_steps.snapshot_env import SnapshotEnvTaskStep


def _deployed(env_id="env-1", instance_id="env-1-abc123", gateway="https://gw-1"):
    return DeployedEnv(
        env_id=env_id,
        env_version=3,
        gateway_url=gateway,
        mcp_url=f"{gateway}/mcp",
        db_web_url=None,
        sandbox_id="sb-1",
        instance_id=instance_id,
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
    first = step._derive_snapshot_id(ctx, "env-1")
    second = step._derive_snapshot_id(ctx, "env-1")
    assert first == second == "snapshot-env-1-65717eb66eeec083"


def test_snapshot_id_override_wins():
    ctx = _ctx(_deployed())
    assert _step(snapshot_id="snap-mine")._derive_snapshot_id(ctx, "env-1") == "snap-mine"


def test_snapshot_id_falls_back_to_random_without_instance_id():
    ctx = _ctx(_deployed(), task_instance_id=None)
    generated = _step()._derive_snapshot_id(ctx, "env-1")
    assert generated.startswith("snapshot-env-1-")


# ── execute wiring ───────────────────────────────────────────────────────────

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
        "id": "snapshot-env-1-65717eb66eeec083",
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
    def __init__(self, store):
        self._store = store

    def get_object_store(self):
        return self._store


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


def _wire_fake_capture(monkeypatch, client):
    store = _FakeObjectStore()
    monkeypatch.setattr(snapshot_mod, "get_config", lambda: _FakeStoreConfig(store))
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


# ── core: all-or-nothing + streaming guard ───────────────────────────────────

class _FakeServiceEnv:
    def __init__(self, name, version=1):
        self.environment_name = name
        self.service_version = version


class _FakeMultiEnv:
    id = "env-1"
    version = 3

    def __init__(self, names):
        self.mcp_server_envs = [_FakeServiceEnv(n) for n in names]


@pytest.mark.asyncio
async def test_core_all_or_nothing_on_partial_failure(monkeypatch):
    async def fake_export(gateway_url, name, tmp_path, timeout):
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
    async def fake_export(gateway_url, name, tmp_path, timeout):
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
    async def fake_export(gateway_url, name, tmp_path, timeout):
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

    async def fake_get_card(base_url):
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

    async def fake_get_card(base_url):
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

    async def fake_get_card(base_url):
        return {}  # no s3-credentials extension -> _push_s3_credentials no-ops

    monkeypatch.setattr(protocol_v1, "supports_v1", fake_supports_v1)
    monkeypatch.setattr(protocol_v1, "get_card", fake_get_card)
    monkeypatch.setattr(protocol_v1, "get_data", fake_get_data)

    with pytest.raises(ValueError, match="neither bytes nor uri"):
        await mod.SnapshotEnvTaskStep._export_environment_to_file(
            "https://gw", "slack", str(tmp_path / "export-tmp"), timeout_seconds=30
        )


def test_enumerate_services_single_mcp_server_env():
    env = _FakeServiceEnv("calendar", version=2)
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
    async def fake_export(gateway_url, name, tmp_path, timeout):
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
