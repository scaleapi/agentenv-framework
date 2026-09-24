"""Unit tests for CollectArtifactsTaskStep.

Focus: the CUA controller path (`env_id` set), which reads files via the
deployed env's gateway (cua_get_file) instead of `docker exec` into the agent
container — required because CUA deliverables live on the desktop VM, not
inside the a2a `agent-api` container.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.task_step.context import DeployedAgent, PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep, _is_url_entry
from agent_env.env.env import DeployedEnv


def _run(coro):
    return asyncio.run(coro)


def _cua_context():
    return TaskStepContext(
        instance_id="inst-1",
        deployed_envs=[DeployedEnv(
            env_id="ubuntu-cua", env_version=1, gateway_url="http://gw",
            mcp_url="http://gw", db_web_url=None, sandbox_id="vm-1",
            metadata={"cua_vm_sandbox_id": "cua-vm-1"},  # CUA marker set by CuaEnv.deploy
        )],
    )


def _non_cua_context():
    """A deployed env that is NOT a CUA env (no cua_vm_* metadata marker)."""
    return TaskStepContext(
        instance_id="inst-1",
        deployed_envs=[DeployedEnv(
            env_id="ubuntu-cua", env_version=1, gateway_url="http://gw",
            mcp_url="http://gw", db_web_url=None, sandbox_id="vm-1",
            metadata={"some_other_env": True},
        )],
    )


class TestSerialization:
    def test_env_id_round_trips(self):
        step = CollectArtifactsTaskStep.from_dict({
            "id": "collect-artifacts", "type": "collect_artifacts",
            "env_id": "ubuntu-cua", "base_path": "/home/docker/Desktop",
            "artifact_paths": ["a.pdf", "b.docx"],
        })
        assert step.env_id == "ubuntu-cua"
        assert step.to_dict()["env_id"] == "ubuntu-cua"

    def test_env_id_defaults_none(self):
        step = CollectArtifactsTaskStep.from_dict({
            "id": "c", "type": "collect_artifacts", "agent_name": "solver",
        })
        assert step.env_id is None

    def test_explicit_agent_name_and_env_id_raises(self):
        try:
            CollectArtifactsTaskStep(
                id="c", version=None, agent_name="solver", env_id="ubuntu-cua",
            )
            assert False, "expected ValueError when both agent_name and env_id are set"
        except ValueError as e:
            assert "not both" in str(e)

    def test_defaulted_agent_name_with_env_id_tolerated(self):
        from agent_env.task_step.task_step import TaskStep
        # Older serialized CUA steps persist DEFAULT_AGENT_NAME alongside env_id;
        # that must not trip the guard.
        step = CollectArtifactsTaskStep(
            id="c", version=None,
            agent_name=TaskStep.DEFAULT_AGENT_NAME, env_id="ubuntu-cua",
        )
        assert step.env_id == "ubuntu-cua"


def _macos_cua_context():
    """A macOS CUA env: gateway_url (CUA MCP server, 18765) plus a distinct
    cua_controller_url (the osworld sidecar, 18768). cua_get_file must go to the
    gateway, not the sidecar — the sidecar's /step only accepts a ScaleCuaAction
    and 500s on a call_tool payload."""
    return TaskStepContext(
        instance_id="inst-1",
        deployed_envs=[DeployedEnv(
            env_id="macos-cua", env_version=1,
            gateway_url="http://gw-18765", mcp_url="http://gw-18765/mcp",
            db_web_url=None, sandbox_id="vm-1",
            metadata={"cua_vm_sandbox_id": "cua-vm-1",
                      "cua_controller_url": "http://ctrl-18768"},
        )],
    )


class TestControllerPath:
    def test_reads_route_to_gateway_mcp_not_controller_sidecar(self):
        """Regression: reads must use gateway_url (CUA MCP server), even when a
        macOS cua_controller_url is present. Routing them to the sidecar (#693)
        made every read 500."""
        step = CollectArtifactsTaskStep(
            id="collect", version=None, env_id="macos-cua",
            artifact_paths=["/Users/admin/Desktop/report.docx"],
        )
        store = MagicMock()
        store.put_object_file.return_value = "s3://bucket/report.docx"
        store.next_version.return_value = 1
        with patch.object(step, "_controller_get_file", new_callable=AsyncMock,
                          return_value=b"data") as get_file, \
             patch("agent_env.artifact.store.get_artifact_store", return_value=store), \
             patch("agent_env.artifact.FileArtifactUniverse") as universe:
            universe.put.return_value = MagicMock(id="inst-1", version=1)
            _run(step.execute(_macos_cua_context()))
        # hit the gateway (18765), NOT the controller sidecar (18768)
        get_file.assert_awaited_once_with("http://gw-18765", "/Users/admin/Desktop/report.docx")

    def test_collects_via_controller_when_env_id_set(self):
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="ubuntu-cua",
            base_path="/home/docker/Desktop", artifact_paths=["report.pdf"],
        )
        store = MagicMock()
        store.put_object_file.return_value = "s3://bucket/report.pdf"
        store.next_version.return_value = 1

        with patch.object(step, "_controller_get_file", new_callable=AsyncMock,
                          return_value=b"%PDF-1.4 data") as get_file, \
             patch("agent_env.artifact.store.get_artifact_store", return_value=store), \
             patch("agent_env.artifact.FileArtifactUniverse") as universe:
            universe.put.return_value = MagicMock(id="inst-1", version=1)
            ctx = _run(step.execute(_cua_context()))

        # fetched the absolute Desktop path through the controller
        get_file.assert_awaited_once_with("http://gw", "/home/docker/Desktop/report.pdf")
        # uploaded and recorded the S3 URI
        assert ctx.metadata["artifacts"] == {"report.pdf": "s3://bucket/report.pdf"}
        assert ctx.metadata["collected_artifacts"]["collect-artifacts"]["artifacts"] == {
            "report.pdf": "s3://bucket/report.pdf"
        }
        store.put_object_file.assert_called_once()

    def test_enumeration_unwraps_the_cua_bash_json_envelope(self):
        """The regression: cua_bash wraps stdout, cua_get_file does not.

        Taken verbatim from a failed run — the whole envelope became one
        "filename" and collection tried to read
        `/home/docker/Desktop/{"error": "", "output": "smoke.txt\\n", ...}`.
        """
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="ubuntu-cua",
            base_path="/home/docker/Desktop",
        )
        envelope = (
            '{"error": "", "output": "smoke.txt\\nnested/report.pdf\\n", '
            '"returncode": 0, "status": "success"}'
        )
        with patch.object(step, "_controller_call", new_callable=AsyncMock,
                          return_value=envelope):
            names = _run(step._controller_list_dir("http://gw"))
        assert names == ["smoke.txt", "nested/report.pdf"]

    def test_enumeration_tolerates_plain_stdout(self):
        """Defensive: a controller that stops wrapping must not break this."""
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="ubuntu-cua",
            base_path="/home/docker/Desktop",
        )
        with patch.object(step, "_controller_call", new_callable=AsyncMock,
                          return_value="smoke.txt\nreport.pdf\n"):
            names = _run(step._controller_list_dir("http://gw"))
        assert names == ["smoke.txt", "report.pdf"]

    def test_enumeration_of_an_empty_desktop_yields_nothing(self):
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="ubuntu-cua",
            base_path="/home/docker/Desktop",
        )
        with patch.object(step, "_controller_call", new_callable=AsyncMock,
                          return_value='{"error": "", "output": "", "returncode": 0}'):
            names = _run(step._controller_list_dir("http://gw"))
        assert names == []

    def test_absolute_paths_bypass_base_path(self):
        """An absolute artifact_paths entry is read verbatim (base_path ignored)
        and stored under its full path (leading slash stripped)."""
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="ubuntu-cua",
            base_path="/home/docker/Desktop",  # should be ignored for the absolute entry
            artifact_paths=["/tmp/output/result.pdf"],
        )
        store = MagicMock()
        store.put_object_file.return_value = "s3://bucket/tmp/output/result.pdf"
        store.next_version.return_value = 1

        with patch.object(step, "_controller_get_file", new_callable=AsyncMock,
                          return_value=b"%PDF data") as get_file, \
             patch("agent_env.artifact.store.get_artifact_store", return_value=store), \
             patch("agent_env.artifact.FileArtifactUniverse") as universe:
            universe.put.return_value = MagicMock(id="inst-1", version=1)
            ctx = _run(step.execute(_cua_context()))

        # read at the exact absolute path — NOT joined to base_path
        get_file.assert_awaited_once_with("http://gw", "/tmp/output/result.pdf")
        # keyed by the full source path (so the UI can show where it came from)
        assert ctx.metadata["artifacts"] == {
            "/tmp/output/result.pdf": "s3://bucket/tmp/output/result.pdf"
        }
        # S3 object name preserves dir structure with the leading slash stripped
        assert store.put_object_file.call_args.kwargs["object_name"] == "tmp/output/result.pdf"

    def test_missing_file_is_skipped_not_fatal(self):
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="ubuntu-cua",
            base_path="/home/docker/Desktop", artifact_paths=["there.pdf", "gone.pdf"],
        )
        store = MagicMock()
        store.put_object_file.return_value = "s3://bucket/there.pdf"
        store.next_version.return_value = 1

        async def fake_get(gateway_url, path):
            if path.endswith("gone.pdf"):
                raise RuntimeError("cua_get_file failed: file not found")
            return b"data"

        with patch.object(step, "_controller_get_file", side_effect=fake_get), \
             patch("agent_env.artifact.store.get_artifact_store", return_value=store), \
             patch("agent_env.artifact.FileArtifactUniverse") as universe:
            universe.put.return_value = MagicMock(id="inst-1", version=1)
            ctx = _run(step.execute(_cua_context()))

        assert ctx.metadata["artifacts"] == {"there.pdf": "s3://bucket/there.pdf"}

    def test_s3_upload_failure_is_skipped_not_fatal(self):
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="ubuntu-cua",
            base_path="/home/docker/Desktop", artifact_paths=["bad.pdf", "ok.pdf"],
        )
        store = MagicMock()
        store.next_version.return_value = 1

        def put(**kw):
            if kw["object_name"] == "bad.pdf":
                raise RuntimeError("S3 timeout")
            return "s3://bucket/ok.pdf"

        store.put_object_file.side_effect = put

        with patch.object(step, "_controller_get_file", new_callable=AsyncMock, return_value=b"data"), \
             patch("agent_env.artifact.store.get_artifact_store", return_value=store), \
             patch("agent_env.artifact.FileArtifactUniverse") as universe:
            universe.put.return_value = MagicMock(id="inst-1", version=1)
            ctx = _run(step.execute(_cua_context()))

        # bad.pdf's S3 failure is non-fatal; ok.pdf still collected
        assert ctx.metadata["artifacts"] == {"ok.pdf": "s3://bucket/ok.pdf"}

    def test_all_resolved_missing_is_fatal(self):
        """Named artifacts that ALL fail to collect -> loud failure, not a silent empty run."""
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="ubuntu-cua",
            base_path="/home/docker/Desktop", artifact_paths=["gone1.pdf", "gone2.pdf"],
        )
        store = MagicMock()
        with patch.object(step, "_controller_get_file", new_callable=AsyncMock,
                          side_effect=RuntimeError("cua_get_file failed: file not found")), \
             patch("agent_env.artifact.store.get_artifact_store", return_value=store):
            try:
                _run(step.execute(_cua_context()))
                assert False, "expected RuntimeError when no named artifact was collected"
            except RuntimeError as e:
                assert "no file_artifact_universe" in str(e)

    def test_url_only_list_collects_nothing_and_does_not_enumerate(self):
        """An explicit list that holds only URLs means the deliverables are those URLs:
        nothing is read from the VM, and base_path is not enumerated as if no list existed."""
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="ubuntu-cua",
            base_path="/home/docker/Desktop",
            artifact_paths=["s3://example-bucket/gold.docx", "https://example.com/report.pdf"],
        )
        store = MagicMock()
        with patch.object(step, "_controller_list_dir", new_callable=AsyncMock) as list_dir, \
             patch.object(step, "_controller_get_file", new_callable=AsyncMock) as get_file, \
             patch("agent_env.artifact.store.get_artifact_store", return_value=store):
            ctx = _run(step.execute(_cua_context()))
        list_dir.assert_not_awaited()
        get_file.assert_not_awaited()
        assert ctx.metadata["artifacts"] == {}
        store.put_object_file.assert_not_called()

    def test_no_list_still_enumerates_base_path(self):
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="ubuntu-cua", base_path="/home/docker/Desktop",
        )
        with patch.object(step, "_controller_list_dir", new_callable=AsyncMock, return_value=[]) as list_dir, \
             patch("agent_env.artifact.store.get_artifact_store", return_value=MagicMock()):
            ctx = _run(step.execute(_cua_context()))
        list_dir.assert_awaited_once_with("http://gw")
        assert ctx.metadata["artifacts"] == {}

    def test_url_only_list_skips_enumeration_on_the_container_path(self):
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, base_path="/app/artifact",
            artifact_paths=["s3://example-bucket/gold.docx"],
        )
        provider = MagicMock()
        provider.close = AsyncMock()
        ctx = TaskStepContext()
        with patch.object(step, "_list_base_directory", new_callable=AsyncMock) as list_dir:
            collected, file_artifacts = _run(step._collect_items(
                provider, MagicMock(), None, step._resolve_items(ctx), ctx, MagicMock(), "aid", 1,
            ))
        list_dir.assert_not_awaited()
        assert (collected, file_artifacts) == ({}, {})
        provider.close.assert_awaited_once()

    def test_collected_but_no_universe_is_fatal(self):
        """Files fetched + uploaded but none register into a universe -> fail at collect, not later at publish."""
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="ubuntu-cua",
            base_path="/home/docker/Desktop", artifact_paths=["present.pdf"],
        )
        store = MagicMock()
        store.put_object_file.return_value = "s3://bucket/present.pdf"
        store.next_version.return_value = 1
        store.put_document.side_effect = RuntimeError("doc write failed")  # registration swallowed -> no universe
        with patch.object(step, "_controller_get_file", new_callable=AsyncMock, return_value=b"data"), \
             patch("agent_env.artifact.store.get_artifact_store", return_value=store):
            try:
                _run(step.execute(_cua_context()))
                assert False, "expected RuntimeError when collected files produce no universe"
            except RuntimeError as e:
                assert "no file_artifact_universe" in str(e)

    def test_param_override_redirects_collection(self):
        """A re-run corrects collect params via user_overrides (no new task version)."""
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="ubuntu-cua",
            base_path="/home/docker/Desktop", artifact_paths=["wrong.pdf"],
        )
        store = MagicMock()
        store.put_object_file.return_value = "s3://bucket/right.pdf"
        store.next_version.return_value = 1
        ctx = _cua_context()
        ctx.metadata["user_overrides"] = {
            "step_params": {"collect-artifacts": {"artifact_paths": ["right.pdf"]}}
        }
        with patch.object(step, "_controller_get_file", new_callable=AsyncMock,
                          return_value=b"data") as get_file, \
             patch("agent_env.artifact.store.get_artifact_store", return_value=store), \
             patch("agent_env.artifact.FileArtifactUniverse") as universe:
            universe.put.return_value = MagicMock(id="inst-1", version=1)
            ctx = _run(step.execute(ctx))
        get_file.assert_awaited_once_with("http://gw", "/home/docker/Desktop/right.pdf")
        assert ctx.metadata["artifacts"] == {"right.pdf": "s3://bucket/right.pdf"}

    def test_unknown_env_id_raises(self):
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="missing-env",
            base_path="/home/docker/Desktop", artifact_paths=["x.pdf"],
        )
        with patch("agent_env.artifact.store.get_artifact_store", return_value=MagicMock()):
            try:
                _run(step.execute(_cua_context()))
                assert False, "expected RuntimeError for unknown env_id"
            except RuntimeError as e:
                assert "missing-env" in str(e)

    def test_non_cua_env_refuses_controller_calls(self):
        """env_id resolving to a non-CUA env must raise before any cua_* call."""
        step = CollectArtifactsTaskStep(
            id="collect-artifacts", version=None, env_id="ubuntu-cua",
            base_path="/home/docker/Desktop", artifact_paths=["x.pdf"],
        )
        with patch.object(step, "_controller_get_file", new_callable=AsyncMock) as get_file, \
             patch("agent_env.artifact.store.get_artifact_store", return_value=MagicMock()):
            try:
                _run(step.execute(_non_cua_context()))
                assert False, "expected RuntimeError for non-CUA env"
            except RuntimeError as e:
                assert "not a CUA environment" in str(e)
        # crucially, no controller tool call was attempted
        get_file.assert_not_awaited()


class TestAgentContainerPath:
    def test_unreachable_sandbox_is_clear_error(self):
        """A re-run targeting an expired/unreachable sandbox fails with a clear message."""
        step = CollectArtifactsTaskStep(
            id="collect", version=None, agent_name="solver",
            base_path="/app/test", artifact_paths=["x.zip"],
        )
        ctx = TaskStepContext(
            instance_id="i",
            deployed_agents=[
                DeployedAgent(agent_name="solver", api_url="http://x", sandbox_id="sb-dead"),
            ],
        )
        sandbox = MagicMock()
        sandbox.mode = "container"
        sandbox.exec_with_output = AsyncMock(return_value=(1, "", "cannot connect"))
        provider = MagicMock()
        provider.get_sandbox = AsyncMock(return_value=sandbox)
        provider.close = AsyncMock()
        with patch(
            "agent_env.providers.sandbox_provider.get_agent_sandbox_provider",
            return_value=provider,
        ), patch("agent_env.artifact.store.get_artifact_store", return_value=MagicMock()):
            try:
                _run(step.execute(ctx))
                assert False, "expected RuntimeError for unreachable sandbox"
            except RuntimeError as e:
                assert "unreachable" in str(e)


def _manifest_ctx(artifacts: dict) -> TaskStepContext:
    # Manifest lives on the producing step's response (keyed by step_id), which collect reads.
    ctx = TaskStepContext()
    ctx.prompt_responses.append(PromptResponse(
        prompt_id="p", response="{}", step_id="m9",
        structured_output={"status": "success", "artifacts": artifacts},
    ))
    return ctx


class TestManifestResolution:
    """`_resolve_items` is pure (context in, (key, source_path, object_name) triples out), so
    these exercise the real resolution logic without a sandbox."""

    def test_manifest_mode_keys_by_logical_name_and_filters(self):
        step = CollectArtifactsTaskStep(
            id="collect", version=None, base_path="/app/swe_atlas",
            manifest_step_id="m9", exclude_basenames=["base_image.tar", "base_image.tar.sha256"],
        )
        ctx = _manifest_ctx({
            "dockerfile":     "/app/swe_atlas/Dockerfile",
            "gold_patch":     "/app/swe_atlas/patches/gold_patch.diff",
            "base_image_tar": "/app/swe_atlas/base_image.tar",        # excluded basename
            "m7_iter_logs":   "/app/swe_atlas/logs/m7_iter_*.json",   # unresolved glob -> skipped
            "coverage_state": "target_reached",                        # literal value, not a path -> skipped
            "harbor_zip":     "/app/swe_atlas/ruff_harbor_v1.zip",
        })
        items = {k: (src, obj) for k, src, obj in step._resolve_items(ctx)}
        assert set(items) == {"dockerfile", "gold_patch", "harbor_zip"}
        # logical name -> (absolute source path, base_path-relative object name)
        assert items["gold_patch"] == ("/app/swe_atlas/patches/gold_patch.diff", "patches/gold_patch.diff")
        assert items["dockerfile"] == ("/app/swe_atlas/Dockerfile", "Dockerfile")

    def test_enumeration_drops_excluded_basenames(self):
        step = CollectArtifactsTaskStep(
            id="collect", version=None, base_path="/app/swe_atlas",
            exclude_basenames=["base_image.tar", "base_image.tar.sha256"],
        )
        enumerated = ["Dockerfile", "patches/gold_patch.diff", "base_image.tar",
                      "base_image.tar.sha256", "245a_harbor_v1.zip"]
        assert step._drop_excluded(enumerated) == ["Dockerfile", "patches/gold_patch.diff", "245a_harbor_v1.zip"]

    def test_manifest_uses_latest_response_when_step_reran(self):
        # On retry/resume a step can append more than once; the latest output wins.
        ctx = TaskStepContext()
        for tag in ("old", "new"):
            ctx.prompt_responses.append(PromptResponse(
                prompt_id="p", response="{}", step_id="m9",
                structured_output={"artifacts": {tag: f"/app/{tag}"}},
            ))
        step = CollectArtifactsTaskStep(id="c", version=None, base_path="/app", manifest_step_id="m9")
        assert {k for k, _, _ in step._resolve_items(ctx)} == {"new"}

    def test_url_scheme_entries_are_skipped_not_mangled(self):
        """URL artifact_paths entries, whatever the scheme, are gold/source URLs
        fetched directly by downstream consumers — collect must skip them, not join
        them to base_path (which produced `/app/artifact/https://…`)."""
        step = CollectArtifactsTaskStep(
            id="collect", version=None, env_id="macos-cua", base_path="/app/artifact",
            artifact_paths=[
                "/Users/admin/Desktop/report.docx",
                "https://example-bucket.s3.amazonaws.com/65cbc42b/1rty2jP6",
                "s3://example-bucket/65cbc42b/foo",
                "vault://65cbc42b/foo#s3/example-bucket",
                "/Users/admin/Desktop/summary.pptx",
            ],
        )
        srcs = {src for _, src, _ in step._resolve_items(TaskStepContext())}
        assert srcs == {"/Users/admin/Desktop/report.docx", "/Users/admin/Desktop/summary.pptx"}
        assert not any("://" in s for s in srcs)

    @pytest.mark.parametrize(
        "entry, is_url",
        [
            ("https://example.com/a", True),
            ("s3://bucket/key", True),
            ("custom+v1.0://x", True),
            ("/abs/path/report.docx", False),
            ("relative/report:final.docx", False),
            ("C:\\Users\\admin\\report.docx", False),
            ("a/b://c", False),
            ("123://x", False),
            ("", False),
        ],
    )
    def test_is_url_entry_matches_a_scheme_prefix_only(self, entry, is_url):
        assert _is_url_entry(entry) is is_url

    def test_manifest_falls_back_to_legacy_metadata_channel(self):
        # Back-compat: pre-DataPart-worker tasks have the manifest in metadata["structured_outputs"].
        ctx = TaskStepContext()
        ctx.metadata["structured_outputs"] = {"m9": {"artifacts": {"dockerfile": "/app/Dockerfile"}}}
        step = CollectArtifactsTaskStep(id="c", version=None, base_path="/app", manifest_step_id="m9")
        assert {k for k, _, _ in step._resolve_items(ctx)} == {"dockerfile"}

    def test_manifest_mode_empty_when_step_declared_nothing(self):
        # No structured output for the step -> no items. execute() turns this into a loud failure.
        step = CollectArtifactsTaskStep(id="c", version=None, manifest_step_id="m9")
        assert step._resolve_items(TaskStepContext()) == []

    def test_legacy_static_mode_unchanged(self):
        step = CollectArtifactsTaskStep(
            id="c", version=None, base_path="/app/artifact", artifact_paths=["Dockerfile", "sub/run.sh"]
        )
        assert step._resolve_items(TaskStepContext()) == [
            ("Dockerfile", "/app/artifact/Dockerfile", "Dockerfile"),
            ("sub/run.sh", "/app/artifact/sub/run.sh", "sub/run.sh"),
        ]

    def test_legacy_seed_mode_unchanged(self):
        step = CollectArtifactsTaskStep(id="c", version=None, base_path="/app/artifact")
        ctx = TaskStepContext()
        ctx.metadata["seed"] = {"expected_artifacts": "a.json, b/c.txt"}
        assert step._resolve_items(ctx) == [
            ("a.json", "/app/artifact/a.json", "a.json"),
            ("b/c.txt", "/app/artifact/b/c.txt", "b/c.txt"),
        ]

    def test_override_wins_over_seed(self):
        # A re-run override of artifact_paths beats a seed-backed list (seed used to win, ignoring it).
        step = CollectArtifactsTaskStep(
            id="c", version=None, base_path="/app", artifacts_key="expected_artifacts"
        )
        ctx = TaskStepContext()
        ctx.metadata["seed"] = {"expected_artifacts": "wrong.zip"}
        ctx.metadata["user_overrides"] = {
            "step_params": {"c": {"artifact_paths": ["right.zip"]}}
        }
        assert [k for k, _, _ in step._resolve_items(ctx)] == ["right.zip"]

    def test_override_string_not_split_into_chars(self):
        step = CollectArtifactsTaskStep(id="c", version=None, base_path="/app")
        ctx = TaskStepContext()
        ctx.metadata["user_overrides"] = {
            "step_params": {"c": {"artifact_paths": "one.zip"}}
        }
        assert [k for k, _, _ in step._resolve_items(ctx)] == ["one.zip"]

    def test_manifest_config_round_trips(self):
        step = CollectArtifactsTaskStep(
            id="c", version=None, manifest_step_id="m9", exclude_basenames=["base_image.tar"]
        )
        rt = CollectArtifactsTaskStep.from_dict(step.to_dict())
        assert rt.manifest_step_id == "m9"
        assert rt.exclude_basenames == {"base_image.tar"}


class TestSandboxContainerPath:
    """`container_name` + `sandbox_name`: collecting from a plain run_docker_container container.

    Before this path existed, a task built from deploy_sandbox -> run_docker_container had no way
    to get files out: `_discover_container` only matches `agent-api` / `a2a-agent-*`, so the agent
    path found nothing and the CUA path refuses a non-CUA env.
    """

    def _context(self):
        from agent_env.task_step.context import DeployedSandbox

        return TaskStepContext(
            instance_id="inst-1",
            deployed_sandboxes=[DeployedSandbox(
                sandbox_name="scraper", sandbox_id="vm-9", sandbox_mode="vm",
                sandbox_type="platform_vm",
            )],
            metadata={"deployed_docker_containers": [
                {"sandbox_name": "scraper", "container_name": "scraper"},
            ]},
        )

    def _step(self, **kw):
        return CollectArtifactsTaskStep(
            id="collect", version=1, sandbox_name="scraper", container_name="scraper",
            base_path="/out", artifact_paths=["page.html"], **kw,
        )

    def test_round_trips(self):
        step = self._step()
        assert step.to_dict()["container_name"] == "scraper"
        assert CollectArtifactsTaskStep.from_dict(step.to_dict()).sandbox_name == "scraper"

    def test_container_name_requires_sandbox_name(self):
        """The lookup key is the pair; a container name alone is not unique across sandboxes."""
        try:
            CollectArtifactsTaskStep(id="c", version=1, container_name="scraper")
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "requires `sandbox_name`" in str(exc)

    def test_container_name_is_exclusive_with_env_id_and_agent_name(self):
        for kw in ({"env_id": "ubuntu-cua"}, {"agent_name": "solver"}):
            try:
                CollectArtifactsTaskStep(id="c", version=1, sandbox_name="s",
                                         container_name="c1", **kw)
                raise AssertionError(f"expected ValueError for {kw}")
            except ValueError as exc:
                assert "exclusive" in str(exc)

    def test_unknown_container_is_a_clear_error(self):
        step = CollectArtifactsTaskStep(id="collect", version=1, sandbox_name="scraper",
                                        container_name="nope", artifact_paths=["a"])
        try:
            _run(step._collect_via_sandbox_container(self._context(), MagicMock(), "aid", 1))
            raise AssertionError("expected RuntimeError")
        except RuntimeError as exc:
            assert "not found on sandbox 'scraper'" in str(exc)

    def test_unknown_sandbox_is_a_clear_error(self):
        ctx = self._context()
        ctx.deployed_sandboxes = []
        try:
            _run(self._step()._collect_via_sandbox_container(ctx, MagicMock(), "aid", 1))
            raise AssertionError("expected RuntimeError")
        except RuntimeError as exc:
            assert "not found in context.deployed_sandboxes" in str(exc)

    def test_collects_through_the_named_container(self):
        step = self._step()
        sandbox = MagicMock()
        provider = MagicMock()
        provider.close = AsyncMock()

        with patch.object(step, "_resolve_live_sandbox", AsyncMock(return_value=sandbox)), \
             patch.object(step, "_collect_items", AsyncMock(return_value=({"page.html": "s3://x"}, {}))) as collect, \
             patch("agent_env.providers.sandbox_provider.build_sandbox_provider",
                   return_value=provider):
            collected, _ = _run(
                step._collect_via_sandbox_container(self._context(), MagicMock(), "aid", 1)
            )

        assert collected == {"page.html": "s3://x"}
        # The explicitly named container is used — never discovery, which would find nothing here.
        assert collect.await_args.args[2] == "scraper"
