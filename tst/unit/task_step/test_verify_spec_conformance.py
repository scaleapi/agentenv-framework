"""Unit tests for the request-side OpenAPI spec-conformance diff.

Covers the pure functions (`normalize_tool_params`, `diff_conformance`) that back
`VerifySpecConformanceTaskStep`. No deployed env, gateway, or network — a fake
Tool object stands in for the live MCP surface.
"""

import logging
from dataclasses import dataclass

from agent_env.task_step.task_steps.mcp_env_validator.verify_spec_conformance import (
    VerifySpecConformanceTaskStep,
    diff_conformance,
    normalize_spec_params,
    normalize_tool_params,
    skipped_result,
)


@dataclass
class FakeTool:
    """Mirror of the mcp Tool shape the diff reads: `.name` + `.inputSchema`."""

    name: str
    inputSchema: dict


def _good_spec() -> dict:
    """A small spec mirroring a real shape: $ref params, a component-schema enum, a
    requestBody, and a path-item-level param — enough to exercise every code path."""
    return {
        "openapi": "3.0.3",
        "paths": {
            "/events": {
                "get": {
                    "operationId": "svc_get_events",
                    "parameters": [
                        {"$ref": "#/components/parameters/Limit"},
                        {
                            "name": "category",
                            "in": "query",
                            "schema": {"$ref": "#/components/schemas/Category"},
                        },
                    ],
                }
            },
            "/events/{event_id}": {
                "get": {
                    "operationId": "svc_get_event",
                    "parameters": [{"$ref": "#/components/parameters/EventId"}],
                }
            },
            # order_id is path-item-level (applies to both ops); expand is GET-level.
            "/orders/{order_id}": {
                "parameters": [
                    {
                        "name": "order_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "get": {
                    "operationId": "svc_get_order",
                    "parameters": [
                        {"name": "expand", "in": "query", "schema": {"type": "boolean"}}
                    ],
                },
                "delete": {"operationId": "svc_delete_order"},
            },
            "/compare": {
                "post": {
                    "operationId": "svc_compare",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CompareRequest"}
                            }
                        },
                    },
                }
            },
        },
        "components": {
            "parameters": {
                "Limit": {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                "EventId": {
                    "name": "event_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            },
            "schemas": {
                "Category": {"type": "string", "enum": ["politics", "crypto", "sports"]},
                "CompareRequest": {
                    "type": "object",
                    "required": ["market_ids"],
                    "properties": {
                        "market_ids": {"type": "array", "items": {"type": "string"}},
                        "normalize": {"type": "boolean"},
                    },
                },
            },
        },
    }


def _good_tools() -> list[FakeTool]:
    """Live tools that conform exactly to `_good_spec()`."""
    return [
        FakeTool("svc_get_events", {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "category": {"type": "string", "enum": ["politics", "crypto", "sports"]},
            },
        }),
        FakeTool("svc_get_event", {
            "type": "object",
            "properties": {"event_id": {"type": "string"}},
            "required": ["event_id"],
        }),
        FakeTool("svc_get_order", {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "expand": {"type": "boolean"}},
            "required": ["order_id"],
        }),
        FakeTool("svc_delete_order", {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        }),
        FakeTool("svc_compare", {
            "type": "object",
            "properties": {
                "market_ids": {"type": "array", "items": {"type": "string"}},
                "normalize": {"type": "boolean"},
            },
            "required": ["market_ids"],
        }),
    ]


class TestConformantCase:
    def test_matched_spec_and_tools_pass(self):
        result = diff_conformance(_good_spec(), _good_tools())
        assert result["passed"], result["findings"]
        assert result["findings"] == []
        assert result["total_tools"] == 5
        assert result["spec_operations"] == 5


class TestToolPresence:
    def test_tool_in_spec_missing_from_server(self):
        tools = [t for t in _good_tools() if t.name != "svc_compare"]
        result = diff_conformance(_good_spec(), tools)
        assert not result["passed"]
        kinds = {(f["kind"], f["tool"]) for f in result["findings"]}
        assert ("missing_tool", "svc_compare") in kinds

    def test_tool_on_server_not_in_spec_is_reported_but_not_blocking(self):
        """Spec is a floor: reported so drift is visible, non-blocking so framework tools
        like `<service>_health` don't fail every server."""
        tools = _good_tools() + [FakeTool("svc_secret", {"type": "object", "properties": {}})]
        result = diff_conformance(_good_spec(), tools)
        assert result["passed"] is True
        assert result["blocking_findings"] == 0
        extra = [f for f in result["findings"] if f["kind"] == "extra_tool"]
        assert [f["tool"] for f in extra] == ["svc_secret"]
        assert extra[0]["blocking"] is False


class TestParamDiff:
    def test_renamed_param(self):
        tools = _good_tools()
        # rename event_id -> id on the live tool
        tools[1] = FakeTool("svc_get_event", {
            "type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"],
        })
        result = diff_conformance(_good_spec(), tools)
        kinds = {(f["kind"], f.get("param")) for f in result["findings"]}
        assert ("missing_param", "event_id") in kinds
        # The rename left a required param the spec never declared, so both halves block.
        assert ("extra_required_param", "id") in kinds
        assert result["passed"] is False

    def test_dropped_enum_value(self):
        tools = _good_tools()
        tools[0] = FakeTool("svc_get_events", {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "category": {"type": "string", "enum": ["politics", "crypto"]},  # sports dropped
            },
        })
        result = diff_conformance(_good_spec(), tools)
        enum_findings = [f for f in result["findings"] if f["kind"] == "enum_mismatch"]
        assert len(enum_findings) == 1
        assert enum_findings[0]["param"] == "category"

    def test_type_mismatch(self):
        tools = _good_tools()
        tools[0].inputSchema["properties"]["limit"] = {"type": "string"}  # spec says integer
        result = diff_conformance(_good_spec(), tools)
        type_findings = [f for f in result["findings"] if f["kind"] == "type_mismatch"]
        assert len(type_findings) == 1
        assert type_findings[0]["param"] == "limit"

    def test_required_mismatch(self):
        tools = _good_tools()
        # event_id is required in spec (path param); drop the required marker on the tool
        tools[1] = FakeTool("svc_get_event", {
            "type": "object", "properties": {"event_id": {"type": "string"}},  # not required
        })
        result = diff_conformance(_good_spec(), tools)
        req_findings = [f for f in result["findings"] if f["kind"] == "required_mismatch"]
        assert len(req_findings) == 1
        assert req_findings[0]["param"] == "event_id"


class TestPathItemLevelParams:
    """Path-item params apply to every operation on the path; dropping them emits a
    bogus `extra_param` — a false block on a conformant server."""

    def test_path_item_param_applies_to_every_operation(self):
        params = normalize_spec_params(_good_spec())
        assert params["svc_get_order"]["order_id"] == {"type": "string", "enum": None, "required": True}
        assert params["svc_delete_order"]["order_id"] == {"type": "string", "enum": None, "required": True}
        # operation-level params still merge in alongside
        assert params["svc_get_order"]["expand"]["type"] == "boolean"

    def test_no_extra_param_finding_for_shared_path_param(self):
        findings = diff_conformance(_good_spec(), _good_tools())["findings"]
        assert [f for f in findings if f.get("param") == "order_id"] == []

    def test_operation_level_overrides_same_named_path_item_param(self):
        spec = _good_spec()
        spec["paths"]["/orders/{order_id}"]["get"]["parameters"].append(
            {"name": "order_id", "in": "path", "required": True, "schema": {"type": "integer"}}
        )
        params = normalize_spec_params(spec)
        assert params["svc_get_order"]["order_id"]["type"] == "integer"
        assert params["svc_delete_order"]["order_id"]["type"] == "string"


class TestRequestBodyNameCollision:
    """A body property must not drop a same-named path param's type or required-ness."""

    def test_body_property_does_not_clobber_path_param(self):
        spec = _good_spec()
        spec["paths"]["/orders/{order_id}"]["get"]["requestBody"] = {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}, "note": {"type": "string"}},
                    }
                }
            }
        }
        params = normalize_spec_params(spec)
        assert params["svc_get_order"]["order_id"]["required"] is True
        assert params["svc_get_order"]["order_id"]["type"] == "string"
        assert params["svc_get_order"]["note"]["required"] is False


class TestNormalizeToolParams:
    def test_anyOf_nullable_resolves_to_underlying_type(self):
        # FastMCP renders Optional[int] as anyOf[{integer}, {null}].
        params = normalize_tool_params({
            "type": "object",
            "properties": {"n": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
        })
        assert params["n"]["type"] == "integer"

    def test_defs_ref_resolves(self):
        params = normalize_tool_params({
            "type": "object",
            "properties": {"c": {"$ref": "#/$defs/Cat"}},
            "$defs": {"Cat": {"type": "string", "enum": ["a", "b"]}},
        })
        assert params["c"]["type"] == "string"
        assert params["c"]["enum"] == ["a", "b"]

    def test_oneOf_and_allOf_resolve_like_anyOf(self):
        # Unresolved, a mismatch behind oneOf/allOf normalizes to type=None and is skipped.
        params = normalize_tool_params({
            "type": "object",
            "properties": {
                "o": {"oneOf": [{"type": "null"}, {"type": "string", "enum": ["x", "y"]}]},
                "a": {"allOf": [{"$ref": "#/$defs/Cat"}]},
            },
            "$defs": {"Cat": {"type": "string", "enum": ["a", "b"]}},
        })
        assert params["o"]["type"] == "string"
        assert params["o"]["enum"] == ["x", "y"]
        assert params["a"]["type"] == "string"

    def test_oneOf_type_mismatch_is_detected(self):
        # spec says limit is an integer; the tool exposes Optional[str] as a oneOf
        spec = _good_spec()
        tools = _good_tools()
        tools[0].inputSchema["properties"]["limit"] = {
            "oneOf": [{"type": "string"}, {"type": "null"}]
        }
        findings = diff_conformance(spec, tools)["findings"]
        assert [f["param"] for f in findings if f["kind"] == "type_mismatch"] == ["limit"]


def _ops_spec(*operation_ids: str) -> dict:
    """Minimal spec declaring one paramless GET per operationId."""
    return {
        "paths": {
            f"/{op}": {"get": {"operationId": op, "parameters": []}}
            for op in operation_ids
        }
    }


class TestServicePrefixNormalization:
    """Tools are namespaced `<service>_<op>`; spec operationIds mostly aren't. Comparing
    verbatim made every tool read as both missing and extra — the motivating regression."""

    def test_prefixed_tool_matches_bare_operation_id(self):
        result = diff_conformance(
            _ops_spec("get_crime_rate"),
            [FakeTool("city_get_crime_rate", {})],
            service_name="city",
        )
        assert result["passed"] is True
        assert result["findings"] == []

    def test_prefixed_spec_matches_bare_tool_name(self):
        # `jira` declares `jira_add_comment` for a tool registered as `add_comment`.
        result = diff_conformance(
            _ops_spec("jira_add_comment"),
            [FakeTool("add_comment", {})],
            service_name="jira",
        )
        assert result["passed"] is True
        assert result["findings"] == []

    def test_name_legitimately_starting_with_service_name_still_matches(self):
        # Symmetric stripping: `city_hall_info` reduces to `hall_info` on both sides.
        result = diff_conformance(
            _ops_spec("city_hall_info"),
            [FakeTool("city_hall_info", {})],
            service_name="city",
        )
        assert result["passed"] is True
        assert result["findings"] == []

    def test_without_service_name_names_are_compared_verbatim(self):
        # Back-compat for callers that don't pass service_name.
        kinds = sorted(
            f["kind"]
            for f in diff_conformance(
                _ops_spec("get_crime_rate"), [FakeTool("city_get_crime_rate", {})]
            )["findings"]
        )
        assert kinds == ["extra_tool", "missing_tool"]

    def test_findings_report_original_not_stripped_names(self):
        findings = diff_conformance(
            _ops_spec("reset"), [FakeTool("city_health", {})], service_name="city"
        )["findings"]
        by_kind = {f["kind"]: f["tool"] for f in findings}
        assert by_kind["missing_tool"] == "reset"
        assert by_kind["extra_tool"] == "city_health"


class TestPrefixStrippingDoesNotOpenAHole:
    """A wrong or undelivered tool must still fail — the objection to floor semantics."""

    def test_a_wrong_name_is_still_caught(self):
        result = diff_conformance(
            _ops_spec("get_crime_rate"),
            [FakeTool("city_get_weather", {})],
            service_name="city",
        )
        assert result["passed"] is False
        assert {f["kind"] for f in result["findings"]} == {"missing_tool", "extra_tool"}

    def test_a_spec_promise_with_no_tool_still_blocks(self):
        result = diff_conformance(
            _ops_spec("get_crime_rate", "reset"),
            [FakeTool("city_get_crime_rate", {})],
            service_name="city",
        )
        assert result["passed"] is False
        assert result["blocking_findings"] == 1
        assert [(f["kind"], f["tool"]) for f in result["findings"]] == [
            ("missing_tool", "reset")
        ]

    def test_framework_tools_alone_do_not_fail_a_server(self):
        # The real-world shape: spec declares its ops, server also registers health +
        # the async-jobs pair. Previously this failed every server.
        result = diff_conformance(
            _ops_spec("get_crime_rate"),
            [
                FakeTool("city_get_crime_rate", {}),
                FakeTool("city_health", {}),
                FakeTool("city_get_job_status", {}),
                FakeTool("city_get_job_result", {}),
            ],
            service_name="city",
        )
        assert result["passed"] is True
        assert result["blocking_findings"] == 0
        assert {f["kind"] for f in result["findings"]} == {"extra_tool"}


class TestAmbiguousTool:
    def test_two_names_reducing_to_one_key_are_reported_and_block(self):
        """`health` + `city_health` would overwrite in the index and hide a tool."""
        result = diff_conformance(
            _ops_spec("health"),
            [FakeTool("health", {}), FakeTool("city_health", {})],
            service_name="city",
        )
        ambiguous = [f for f in result["findings"] if f["kind"] == "ambiguous_tool"]
        assert len(ambiguous) == 1
        assert ambiguous[0]["blocking"] is True
        assert result["passed"] is False
        # Names the prefix that caused the collision, without pinning the phrasing.
        assert "city_" in ambiguous[0]["detail"]

    def test_duplicate_names_do_not_blame_a_prefix_that_was_never_stripped(self):
        """Without a service_name nothing is stripped, so a collision is a literal duplicate
        — the detail must not read "the 'None_' prefix"."""
        result = diff_conformance(
            _ops_spec("get_x"), [FakeTool("get_x", {}), FakeTool("get_x", {})]
        )
        detail = next(f["detail"] for f in result["findings"] if f["kind"] == "ambiguous_tool")
        # Only the wrong wording is pinned; the message itself is free to be reworded.
        assert "prefix" not in detail and "None" not in detail

    def test_spec_operations_counts_raw_ops_not_the_collapsed_index(self):
        # Both operationIds reduce to 'health'; reporting 1 would under-count exactly when
        # the ambiguous_tool finding says something is wrong.
        result = diff_conformance(
            _ops_spec("health", "city_health"), [FakeTool("city_health", {})],
            service_name="city",
        )
        assert result["spec_operations"] == 2


class TestBlockingClassification:
    def test_every_finding_is_labelled_and_counted(self):
        spec = _ops_spec("get_crime_rate", "reset")
        tools = [FakeTool("city_get_crime_rate", {}), FakeTool("city_health", {})]
        result = diff_conformance(spec, tools, service_name="city")

        by_kind = {f["kind"]: f["blocking"] for f in result["findings"]}
        assert by_kind == {"missing_tool": True, "extra_tool": False}
        assert result["blocking_findings"] == 1
        assert result["passed"] is False

    def test_extra_param_is_informational_while_missing_param_blocks(self):
        spec = {
            "paths": {
                "/thing": {
                    "get": {
                        "operationId": "get_thing",
                        "parameters": [
                            {"name": "wanted", "in": "query", "schema": {"type": "string"}}
                        ],
                    }
                }
            }
        }
        tools = [FakeTool("svc_get_thing", {
            "type": "object",
            "properties": {"unexpected": {"type": "string"}},
        })]
        result = diff_conformance(spec, tools, service_name="svc")
        by_kind = {f["kind"]: f["blocking"] for f in result["findings"]}
        assert by_kind == {"missing_param": True, "extra_param": False}
        assert result["blocking_findings"] == 1
        assert result["passed"] is False

    def test_undeclared_required_param_blocks(self):
        """The one extra that isn't free: a caller reading only the spec omits it, so the
        call fails. Optional in the same position stays informational."""
        spec = {"paths": {"/thing": {"get": {"operationId": "get_thing", "parameters": []}}}}

        def _result(required: list[str]) -> dict:
            tools = [FakeTool("svc_get_thing", {
                "type": "object",
                "properties": {"undeclared": {"type": "string"}},
                "required": required,
            })]
            return diff_conformance(spec, tools, service_name="svc")

        blocked = _result(["undeclared"])
        assert blocked["passed"] is False
        assert blocked["blocking_findings"] == 1
        assert [(f["kind"], f["param"]) for f in blocked["findings"]] == [
            ("extra_required_param", "undeclared")
        ]

        allowed = _result([])
        assert allowed["passed"] is True
        assert [f["kind"] for f in allowed["findings"]] == ["extra_param"]


class TestSkippedResult:
    def test_skipped_is_not_a_pass_but_is_marked_skipped(self):
        result = skipped_result("no OpenAPI spec served")
        assert result["skipped"] is True
        assert result["passed"] is False
        assert result["reason"] == "no OpenAPI spec served"

    def test_skipped_exposes_blocking_findings_key(self):
        # Consumers (CLI display, aggregator) read blocking_findings unconditionally.
        assert skipped_result("nothing to check")["blocking_findings"] == 0


class TestRecordLogging:
    """`_record` must log drift whenever there is any, not only on a failed verdict.

    Keying the log off `passed` silenced the normal case: every server reports its
    framework tools as informational `extra_*`, so the verdict passes and the drift went
    unlogged. Level tracks severity so expected framework-tool drift doesn't warn on every
    run.
    """

    def _record(self, result, caplog):
        """`_record` reads no instance state, so call it unbound rather than build a step
        (whose base `__init__` needs a store)."""
        class FakeEnv:
            def __init__(self): self.merged = {}
            def merge_metadata(self, d): self.merged.update(d)

        class FakeContext:
            def __init__(self): self.metadata = {}

        env, context = FakeEnv(), FakeContext()
        with caplog.at_level(logging.DEBUG):
            VerifySpecConformanceTaskStep._record(None, context, env, "city", result)
        return env, context, caplog.records

    def test_informational_only_drift_is_logged_at_info(self, caplog):
        result = {
            "passed": True, "blocking_findings": 0,
            "findings": [{"kind": "extra_tool", "tool": "city_health", "blocking": False}],
        }
        _, _, records = self._record(result, caplog)
        drift = [r for r in records if "drift" in r.message]
        assert len(drift) == 1
        assert drift[0].levelno == logging.INFO
        assert "city_health" in drift[0].message

    def test_blocking_drift_warns(self, caplog):
        result = {
            "passed": False, "blocking_findings": 1,
            "findings": [{"kind": "missing_tool", "tool": "reset", "blocking": True}],
        }
        _, _, records = self._record(result, caplog)
        drift = [r for r in records if "drift" in r.message]
        assert len(drift) == 1
        assert drift[0].levelno == logging.WARNING

    def test_a_clean_server_logs_no_drift(self, caplog):
        result = {"passed": True, "blocking_findings": 0, "findings": []}
        _, _, records = self._record(result, caplog)
        assert [r for r in records if "drift" in r.message] == []

    def test_skipped_logs_its_reason_and_no_drift(self, caplog):
        result = skipped_result("no spec served")
        _, _, records = self._record(result, caplog)
        assert any("skipped" in r.message and "no spec served" in r.message for r in records)
        assert [r for r in records if "drift" in r.message] == []

    def test_verdict_is_persisted_either_way(self, caplog):
        result = {
            "passed": True, "blocking_findings": 0,
            "findings": [{"kind": "extra_tool", "tool": "city_health", "blocking": False}],
        }
        env, context, _ = self._record(result, caplog)
        assert env.merged == {"mcp_spec_conformance": result}
        assert context.metadata["verifications"]["spec_conformance"] is result
