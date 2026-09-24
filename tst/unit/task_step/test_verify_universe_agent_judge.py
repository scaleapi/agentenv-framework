"""Unit tests for the file-based universe agent-judge.

Covers the pure prompt/rubric builders and the FileArtifactUniverse-emit helper on the round-trip
step. The helper test mocks the artifact store so it runs with no external services, and asserts
the transport is **faithful** (original/export1 reference the existing FileArtifacts verbatim;
export2 is dumped losslessly, unicode preserved).
"""

import json

from agent_env.artifact import FileArtifact, FileArtifactUniverse
from agent_env.task_step import VerifyUniverseLoadExportRoundtripStep
from agent_env.task_step.task_steps.multienv_validator.verify_universe_agent_judge import (
    CROSS_SERVICE_KEY,
    JUDGE_SYSTEM_PROMPT,
    UNIVERSES_DIR,
    apply_judge_verdict,
    build_criteria,
    build_user_prompt,
    judge_issues_from_verdict,
)


class TestBuildCriteria:
    def test_one_criterion_per_service_plus_cross_cutting(self):
        crit = build_criteria(["slack", "oracle_gl", "airtable"])
        service_ids = [c["id"] for c in crit if c["id"].startswith("service_preserved__")]
        assert service_ids == [
            "service_preserved__slack",
            "service_preserved__oracle_gl",
            "service_preserved__airtable",
        ]
        # 3 per-service + the 7 cross-cutting FLAG criteria
        assert len(crit) == 10
        assert any(c["id"] == "record_preservation" for c in crit)
        assert any(c["id"] == "value_fidelity" for c in crit)

    def test_every_criterion_has_required_keys(self):
        for c in build_criteria(["slack"]):
            assert set(c) >= {"id", "title", "rubric_category", "rubric_target"}

    def test_empty_service_list_yields_only_cross_cutting(self):
        crit = build_criteria([])
        assert len(crit) == 7
        assert all(not c["id"].startswith("service_preserved__") for c in crit)


class TestPrompts:
    def test_system_prompt_has_ignore_and_flag_sections(self):
        assert "IGNORE" in JUDGE_SYSTEM_PROMPT
        assert "FLAG" in JUDGE_SYSTEM_PROMPT
        assert UNIVERSES_DIR in JUDGE_SYSTEM_PROMPT

    def test_user_prompt_points_at_universes_dir(self):
        assert UNIVERSES_DIR in build_user_prompt()
        assert "original" in build_user_prompt() and "export_1" in build_user_prompt()


class TestFauId:
    def test_deterministic(self):
        a = VerifyUniverseLoadExportRoundtripStep.file_artifact_universe_id("env", 3, "uni", 7)
        b = VerifyUniverseLoadExportRoundtripStep.file_artifact_universe_id("env", 3, "uni", 7)
        assert a == b == "validate-env-v3-uni-v7-fau"

    def test_emit_flag_round_trips_through_serialization(self):
        s = VerifyUniverseLoadExportRoundtripStep(
            id="x", version=None, env_id="e", universe_artifact_id="u", emit_file_artifact_universe=True
        )
        assert VerifyUniverseLoadExportRoundtripStep.from_dict(s.to_dict()).emit_file_artifact_universe is True

    def test_emit_flag_defaults_false(self):
        s = VerifyUniverseLoadExportRoundtripStep(id="x", version=None, env_id="e", universe_artifact_id="u")
        assert s.emit_file_artifact_universe is False
        # back-compat: a serialized doc without the field deserializes to False
        d = s.to_dict()
        del d["emit_file_artifact_universe"]
        assert VerifyUniverseLoadExportRoundtripStep.from_dict(d).emit_file_artifact_universe is False


class TestJudgeIssuesFromVerdict:
    def test_flagged_per_service_criterion_becomes_critical_issue_on_that_service(self):
        verdict = {"results": [
            {"id": "service_preserved__oracle_gl", "result": False, "justification": "product_code dropped"},
            {"id": "service_preserved__slack", "result": True, "justification": "ok"},
        ]}
        grouped = judge_issues_from_verdict(verdict)
        assert set(grouped) == {"oracle_gl"}  # passing service produces no issue
        issue = grouped["oracle_gl"][0]
        assert issue["entity"] == "oracle_gl" and issue["critical"] is True
        assert issue["type"] == "agent_judge" and issue["phase"] == "agent_judge"
        assert issue["detail"] == "product_code dropped"

    def test_cross_cutting_criterion_buckets_under_cross_service(self):
        verdict = {"results": [{"id": "referential_integrity", "result": False, "justification": "orphaned ref"}]}
        grouped = judge_issues_from_verdict(verdict)
        assert set(grouped) == {CROSS_SERVICE_KEY}
        assert grouped[CROSS_SERVICE_KEY][0]["field"] == "referential_integrity"

    def test_score_below_one_flags_when_no_bool_result(self):
        verdict = {"results": [
            {"id": "service_preserved__slack", "score": 0.0},
            {"id": "service_preserved__email", "score": 1.0},
        ]}
        grouped = judge_issues_from_verdict(verdict)
        assert set(grouped) == {"slack"}

    def test_all_passing_yields_no_issues(self):
        verdict = {"results": [
            {"id": "service_preserved__slack", "result": True},
            {"id": "value_fidelity", "result": True},
        ]}
        assert judge_issues_from_verdict(verdict) == {}

    def test_missing_justification_gets_a_default_detail(self):
        grouped = judge_issues_from_verdict({"results": [{"id": "numeric_precision", "result": False}]})
        assert grouped[CROSS_SERVICE_KEY][0]["detail"]


class TestApplyJudgeVerdict:
    def _pv_compat(self):
        # A clean programmatic result: two services, both compatible, one with a benign issue.
        return {
            "compatible": True,
            "services": {
                "slack": {"compatible": True, "issues": [{"entity": "slack", "field": "x", "type": "added_field", "phase": "export", "critical": False, "detail": "benign"}]},
                "oracle_gl": {"compatible": True, "issues": []},
            },
        }

    def test_clean_verdict_leaves_compatible_and_keeps_pv_issues(self):
        compat = self._pv_compat()
        out = apply_judge_verdict(compat, {"results": [{"id": "value_fidelity", "result": True}], "score": 1.0})
        assert out is compat  # mutated in place
        assert out["compatible"] is True
        assert CROSS_SERVICE_KEY not in out["services"]
        assert len(out["services"]["slack"]["issues"]) == 1  # PV issue preserved
        assert out["agent_judge"]["score"] == 1.0  # raw verdict stored

    def test_per_service_flag_gates_that_service_and_overall(self):
        compat = self._pv_compat()
        apply_judge_verdict(compat, {"results": [{"id": "service_preserved__oracle_gl", "result": False, "justification": "lost product_code"}]})
        svc = compat["services"]["oracle_gl"]
        assert svc["compatible"] is False
        assert svc["issues"][0]["type"] == "agent_judge" and svc["issues"][0]["critical"] is True
        assert compat["compatible"] is False
        # untouched service stays compatible
        assert compat["services"]["slack"]["compatible"] is True

    def test_cross_cutting_flag_adds_cross_service_bucket_and_gates(self):
        compat = self._pv_compat()
        apply_judge_verdict(compat, {"results": [{"id": "referential_integrity", "result": False, "justification": "dangling ref"}]})
        assert compat["services"][CROSS_SERVICE_KEY]["compatible"] is False
        assert compat["compatible"] is False

    def test_appends_to_existing_pv_issues_not_replace(self):
        compat = self._pv_compat()
        apply_judge_verdict(compat, {"results": [{"id": "service_preserved__slack", "result": False, "justification": "msg dropped"}]})
        types = [i["type"] for i in compat["services"]["slack"]["issues"]]
        assert types == ["added_field", "agent_judge"]  # PV issue kept, judge issue appended


class _FakeFileArtifact:
    """Sentinel standing in for a persisted FileArtifact."""

    def __init__(self, id):
        self.id = id


class _FakeServiceArtifact:
    def __init__(self, environment_name):
        self.environment_name = environment_name
        self._fa = _FakeFileArtifact(f"fa-{environment_name}")

    def get_file_artifact(self):
        return self._fa


class TestCreateFileArtifactUniverse:
    def test_bundles_three_dirs_faithfully(self, monkeypatch):
        # Capture what the helper hands to the store, plus the bytes it writes for export2.
        put_calls = {}
        dumped = {}

        def fake_fa_put(id, description, file_path):
            with open(file_path, encoding="utf-8") as f:
                dumped[id] = f.read()
            return _FakeFileArtifact(id)

        def fake_fau_put(id, *, file_artifacts):
            put_calls["id"] = id
            put_calls["file_artifacts"] = file_artifacts
            return _FakeFileArtifact(id)

        monkeypatch.setattr(FileArtifact, "put", staticmethod(fake_fa_put))
        monkeypatch.setattr(FileArtifactUniverse, "put", staticmethod(fake_fau_put))

        step = VerifyUniverseLoadExportRoundtripStep(
            id="s", version=None, env_id="multi-x", universe_artifact_id="uni", emit_file_artifact_universe=True
        )

        orig_sas = [_FakeServiceArtifact("slack"), _FakeServiceArtifact("oracle_gl")]

        class _FakeExportedUniverse:
            def get_environment_artifacts(self):
                return [_FakeServiceArtifact("slack"), _FakeServiceArtifact("oracle_gl")]

        # unicode + float to prove lossless, ensure_ascii=False dump
        export2_raw = {
            "slack": {"messages": [{"body": "héllo", "ts": 1772461920.0}]},
            "oracle_gl": {"lines": [{"amount": 1500000001.5, "product_code": "OACCT"}]},
        }

        fau = step._create_file_artifact_universe(orig_sas, _FakeExportedUniverse(), export2_raw, env_version=3, universe_version=7)

        assert fau.id == "validate-multi-x-v3-uni-v7-fau"
        assert put_calls["id"] == "validate-multi-x-v3-uni-v7-fau"

        fa_map = put_calls["file_artifacts"]
        assert set(fa_map) == {
            "original/slack.json", "original/oracle_gl.json",
            "export_1/slack.json", "export_1/oracle_gl.json",
            "export_2/slack.json", "export_2/oracle_gl.json",
        }
        # original/export_1 reference the EXISTING FileArtifacts verbatim (no re-creation)
        assert fa_map["original/slack.json"] is orig_sas[0].get_file_artifact()
        assert fa_map["export_1/slack.json"].id == "fa-slack"

        # export_2 dumped losslessly: re-parses to the same object, unicode kept literal
        slack_dumped = dumped["validate-multi-x-v3-uni-v7-export2-slack"]
        assert json.loads(slack_dumped) == export2_raw["slack"]
        assert "héllo" in slack_dumped  # ensure_ascii=False
        assert json.loads(dumped["validate-multi-x-v3-uni-v7-export2-oracle_gl"]) == export2_raw["oracle_gl"]


class TestFrontendContract:
    """Pin the shape of the merged UNIVERSE_COMPATIBILITY doc the hub frontend renders.

    Contract source: the hub frontend's env-detail page component, which
    reads `result.data.{compatible, services}` and, per service, `{compatible, issues[]}` where each
    issue is `{entity, field, type, detail, phase, critical}`. If a producer-side rename drifts from
    that interface this test fails, forcing a conscious frontend sync. Mirrors what the UI-seed script
    writes (it builds its doc via the same apply_judge_verdict)."""

    # exact set the frontend reads off every issue
    FRONTEND_ISSUE_FIELDS = {"entity", "field", "type", "phase", "critical", "detail"}

    def _representative_merged_doc(self):
        # PV result with a critical, a benign non-critical, and a clean service (mirrors the seed)
        compat = {
            "compatible": False,
            "services": {
                "airtable": {"compatible": False, "issues": [
                    {"entity": "tables", "field": "views", "type": "value_mismatch",
                     "phase": "load", "critical": True, "detail": "Values differ (5 occurrences)"}]},
                "slack": {"compatible": True, "issues": [
                    {"entity": "messages", "field": "reactions", "type": "added_field",
                     "phase": "export", "critical": False, "detail": "server enrichment"}]},
                "email": {"compatible": True, "issues": []},
            },
        }
        verdict = {"results": [
            {"id": "service_preserved__records_vault", "result": False, "score": 0.0, "justification": "entity_id nulled"},
            {"id": "referential_integrity", "result": False, "score": 0.0, "justification": "dangling refs"},
            {"id": "service_preserved__slack", "result": True, "score": 1.0, "justification": "ok"},
        ], "score": 0.0}
        apply_judge_verdict(compat, verdict)
        return compat

    def test_top_level_shape(self):
        doc = self._representative_merged_doc()
        assert {"compatible", "services"} <= set(doc)
        assert isinstance(doc["compatible"], bool)
        assert "agent_judge" in doc  # raw verdict retained for audit

    def test_every_service_and_issue_matches_frontend_interface(self):
        doc = self._representative_merged_doc()
        # includes both PV services and the judge-created "(cross-service)" + records_vault buckets
        assert {"airtable", "slack", "email", "records_vault", CROSS_SERVICE_KEY} <= set(doc["services"])
        for name, svc in doc["services"].items():
            assert {"compatible", "issues"} <= set(svc), f"service {name!r} missing keys"
            assert isinstance(svc["compatible"], bool)
            for issue in svc["issues"]:
                assert self.FRONTEND_ISSUE_FIELDS <= set(issue), f"{name} issue missing keys: {set(issue)}"
                assert isinstance(issue["critical"], bool)
                assert isinstance(issue["entity"], str) and isinstance(issue["detail"], str)

    def test_judge_and_pv_issues_coexist_in_same_shape(self):
        # a PV issue and a judge issue must be indistinguishable structurally (only type/phase differ)
        doc = self._representative_merged_doc()
        pv_issue = doc["services"]["slack"]["issues"][0]
        judge_issue = doc["services"][CROSS_SERVICE_KEY]["issues"][0]
        assert set(pv_issue) >= self.FRONTEND_ISSUE_FIELDS
        assert set(judge_issue) >= self.FRONTEND_ISSUE_FIELDS
        assert judge_issue["type"] == "agent_judge" and judge_issue["phase"] == "agent_judge"
