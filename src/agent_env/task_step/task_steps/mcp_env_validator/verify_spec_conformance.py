"""Verify a deployed MCP server conforms to its OpenAPI spec (request side).

Diffs the baked spec (served at ``/openapi.yaml``) against the live ``list_tools()``
surface: tool presence, parameter names, types, enums, required-ness. Response shape
is out of scope.

The spec is a FLOOR: what it promises must match, but a server may offer more
(``extra_tool`` / ``extra_param`` are reported, not blocking) — that keeps framework tools
like ``<service>_health`` from failing every server. The exception is an undeclared param
the tool *requires*, which no spec-following caller can satisfy.

Record-only, and not in ``DEFAULT_REQUIRED_GATES``: findings persist under
``mcp_spec_conformance`` so drift stays measurable, and a pipeline can opt back into
enforcement by passing ``spec_conformance`` in ``ValidationGateAggregatorStep``'s
``required_gates`` (a step arg, not env metadata). No baked spec = ``skipped``. Execution
errors (unreachable MCP) still raise.

Keep the diff logic in sync with the universe-generation pipeline's
``synthetic_mcp_server_generation/validation/spec_conformance.py`` (same diff at generation
time); tool shape differs — MCP objects here, dicts there.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

import httpx
import yaml
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agent_env.env import legacy_protocol
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.mcp_env_validator.verify_mcp_tool_schema import (
    resolve_defs_ref,
)

logger = logging.getLogger(__name__)

# OpenAPI HTTP methods that can carry an operationId → MCP tool.
_HTTP_METHODS = ("get", "post", "put", "delete", "patch", "head", "options", "trace")

# Shared by the spec fetch and the MCP tool-list read; httpx's default is too short.
_HTTP_TIMEOUT_S = 30.0


def _resolve_openapi_ref(node: Any, spec: dict) -> Any:
    """Follow ``#/components/...`` ``$ref`` pointers against the whole spec.

    Sibling keys on the referencing node (e.g. ``description``) override the
    resolved target. Bounded against ``$ref`` cycles; returns the node unchanged
    if a pointer can't be resolved.
    """
    seen: set[str] = set()
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen:
            return node
        seen.add(ref)
        target: Any = spec
        for part in ref[2:].split("/"):
            if not isinstance(target, dict) or part not in target:
                return node
            target = target[part]
        siblings = {k: v for k, v in node.items() if k != "$ref"}
        node = {**target, **siblings} if isinstance(target, dict) else target
    return node


def _schema_type_and_enum(schema: Any, spec: dict) -> tuple[Optional[str], Optional[list]]:
    """Extract a comparable (type, sorted-enum) pair from an OpenAPI schema node.

    Handles ``$ref`` and ``anyOf``/``oneOf``/``allOf`` (first non-null typed variant).
    """
    schema = _resolve_openapi_ref(schema, spec)
    if not isinstance(schema, dict):
        return None, None
    typ = schema.get("type")
    enum = schema.get("enum")
    if typ is None:
        for key in ("anyOf", "oneOf", "allOf"):
            for variant in schema.get(key, []) or []:
                v = _resolve_openapi_ref(variant, spec)
                if isinstance(v, dict) and v.get("type") and v.get("type") != "null":
                    typ = v.get("type")
                    enum = enum or v.get("enum")
                    break
            if typ:
                break
    return typ, (sorted(str(e) for e in enum) if enum else None)


def normalize_spec_params(spec: dict) -> dict[str, dict[str, dict]]:
    """Map each ``operationId`` → ``{param_name: {type, enum, required}}``.

    Collects path-item-level and operation-level ``parameters`` plus ``requestBody``
    JSON properties.
    """
    result: dict[str, dict[str, dict]] = {}
    paths = spec.get("paths", {}) or {}
    for _path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        # Path-item-level parameters apply to every operation on the path.
        shared_params = methods.get("parameters", []) or []
        for method, op in methods.items():
            if method.lower() not in _HTTP_METHODS or not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            if not op_id:
                continue
            params: dict[str, dict] = {}

            # Operation-level last so it overrides a same-named path-item parameter.
            for p in [*shared_params, *(op.get("parameters", []) or [])]:
                p = _resolve_openapi_ref(p, spec)
                if not isinstance(p, dict) or "name" not in p:
                    continue
                typ, enum = _schema_type_and_enum(p.get("schema", {}) or {}, spec)
                params[p["name"]] = {
                    "type": typ,
                    "enum": enum,
                    "required": bool(p.get("required", False)),
                }

            body = op.get("requestBody")
            if isinstance(body, dict):
                body = _resolve_openapi_ref(body, spec)
                schema = (
                    (body.get("content", {}) or {})
                    .get("application/json", {})
                    .get("schema", {})
                )
                schema = _resolve_openapi_ref(schema, spec) if schema else {}
                if isinstance(schema, dict):
                    req = set(schema.get("required", []) or [])
                    for name, psch in (schema.get("properties", {}) or {}).items():
                        typ, enum = _schema_type_and_enum(psch or {}, spec)
                        entry = {"type": typ, "enum": enum, "required": name in req}
                        # A parameter and a body property can share a name: keep the
                        # parameter's type/enum, require if either side requires.
                        existing = params.get(name)
                        if existing is not None:
                            entry["type"] = existing["type"] if existing["type"] is not None else typ
                            entry["enum"] = existing["enum"] if existing["enum"] is not None else enum
                            entry["required"] = existing["required"] or entry["required"]
                        params[name] = entry

            result[op_id] = params
    return result


def normalize_tool_params(input_schema: dict) -> dict[str, dict]:
    """Map a live MCP tool's ``inputSchema`` → ``{param_name: {type, enum, required}}``.

    Reuses ``resolve_defs_ref`` for ``#/$defs/``; handles ``anyOf``/``oneOf``/``allOf``
    symmetrically with the spec side (``_schema_type_and_enum``).
    """
    input_schema = input_schema or {}
    props = input_schema.get("properties", {}) or {}
    required = set(input_schema.get("required", []) or [])
    out: dict[str, dict] = {}
    for name, psch in props.items():
        if not isinstance(psch, dict):
            out[name] = {"type": None, "enum": None, "required": name in required}
            continue
        resolved = resolve_defs_ref(psch, input_schema)
        typ = resolved.get("type")
        enum = resolved.get("enum")
        if typ is None:
            for key in ("anyOf", "oneOf", "allOf"):
                for v in resolved.get(key, []) or []:
                    rv = (
                        resolve_defs_ref(v, input_schema)
                        if isinstance(v, dict)
                        else v
                    )
                    if isinstance(rv, dict) and rv.get("type") and rv.get("type") != "null":
                        typ = rv.get("type")
                        enum = enum or rv.get("enum")
                        break
                if typ:
                    break
        out[name] = {
            "type": typ,
            "enum": sorted(str(e) for e in enum) if enum else None,
            "required": name in required,
        }
    return out


# Findings that break a spec-following caller. `extra_tool`/`extra_param` are omitted on
# purpose: a server offering more than it documents breaks nobody, so it's informational.
BLOCKING_FINDING_KINDS = frozenset({
    "missing_tool",
    "missing_param",
    "type_mismatch",
    "enum_mismatch",
    "required_mismatch",
    "ambiguous_tool",  # can't tell two tools apart -> could be masking a missing one
    # The one extra that isn't free: a caller reading only the spec omits an undeclared
    # required param, so the call fails. Undeclared-and-optional stays informational.
    "extra_required_param",
})


def strip_service_prefix(name: str, service_name: Optional[str]) -> str:
    """Drop a leading ``<service_name>_`` to exclude the prefix from name comparison.

    Tools are namespaced (one gateway fronts every service); spec ``operationId``s mostly
    omit it. Stripped on BOTH sides because the prefix sits on the tool for some servers
    (``city_get_crime_rate`` vs ``get_crime_rate``) and on the spec for others (``jira``
    declares ``jira_add_comment`` for tool ``add_comment``). Symmetric stripping keeps
    ``city_hall_info`` matching itself. Params/types/enums still compared strictly.
    """
    if not service_name:
        return name
    prefix = f"{service_name}_"
    return name[len(prefix):] if name.startswith(prefix) else name


def _index_by_normalized(
    pairs: list[tuple[str, dict]], service_name: Optional[str]
) -> tuple[dict[str, tuple[str, dict]], list[tuple[str, str, str]]]:
    """Key ``(original_name, params)`` pairs by prefix-stripped name.

    Returns ``(index, collisions)``. On a collision (``health`` and ``<service>_health``)
    the later name takes the slot and the earlier one drops out of the diff, so the pair is
    also returned and reported as a blocking ``ambiguous_tool`` rather than lost silently.
    """
    index: dict[str, tuple[str, dict]] = {}
    collisions: list[tuple[str, str, str]] = []
    for original, params in pairs:
        key = strip_service_prefix(original, service_name)
        if key in index:
            collisions.append((key, index[key][0], original))
        index[key] = (original, params)
    return index, collisions


def _collision_reason(key: str, service_name: Optional[str]) -> str:
    """Why two names collided. Without a ``service_name`` nothing was stripped, so they are
    literal duplicates — don't claim a prefix was involved."""
    if not service_name:
        return "are declared more than once"
    return f"both reduce to '{key}' after stripping the '{service_name}_' prefix"


def diff_conformance(
    spec: dict, tools: list, service_name: Optional[str] = None
) -> dict[str, Any]:
    """Request-side conformance diff between an OpenAPI spec and live MCP tools.

    ``tools`` is a list of MCP SDK tool objects. ``service_name`` excludes the
    ``<service>_`` prefix from name comparison; omit it to compare verbatim.

    Returns ``{passed, total_tools, spec_operations, findings, blocking_findings}``. Kinds:
    ``missing_tool``, ``extra_tool``, ``ambiguous_tool``, ``missing_param``,
    ``extra_param``, ``extra_required_param``, ``type_mismatch``, ``enum_mismatch``,
    ``required_mismatch``. Each finding carries ``blocking`` (see
    :data:`BLOCKING_FINDING_KINDS`); ``passed`` reflects blocking findings only. Findings
    name the ORIGINAL, unstripped tool.
    """
    spec_params = normalize_spec_params(spec)
    spec_ops, spec_collisions = _index_by_normalized(list(spec_params.items()), service_name)
    live, live_collisions = _index_by_normalized(
        [(t.name, normalize_tool_params(t.inputSchema or {})) for t in tools],
        service_name,
    )

    findings: list[dict] = []
    spec_names, live_names = set(spec_ops), set(live)

    for key, first, second in spec_collisions:
        findings.append({
            "kind": "ambiguous_tool", "tool": second,
            "detail": f"spec operationIds '{first}' and '{second}' "
                      f"{_collision_reason(key, service_name)}",
        })
    for key, first, second in live_collisions:
        findings.append({
            "kind": "ambiguous_tool", "tool": second,
            "detail": f"MCP tools '{first}' and '{second}' "
                      f"{_collision_reason(key, service_name)}",
        })

    for key in sorted(spec_names - live_names):
        name = spec_ops[key][0]
        findings.append({
            "kind": "missing_tool", "tool": name,
            "detail": f"operationId '{name}' in spec has no matching MCP tool",
        })
    for key in sorted(live_names - spec_names):
        name = live[key][0]
        findings.append({
            "kind": "extra_tool", "tool": name,
            "detail": f"MCP tool '{name}' is not declared in the spec",
        })

    for key in sorted(spec_names & live_names):
        _spec_name, sp = spec_ops[key]
        # Report against the live tool name — that's what a caller actually invokes.
        name, lv = live[key]
        for pname in sorted(set(sp) - set(lv)):
            findings.append({
                "kind": "missing_param", "tool": name, "param": pname,
                "detail": f"spec param '{pname}' absent from tool inputSchema",
            })
        for pname in sorted(set(lv) - set(sp)):
            required = lv[pname]["required"]
            findings.append({
                "kind": "extra_required_param" if required else "extra_param",
                "tool": name, "param": pname,
                "detail": f"tool param '{pname}' not declared in spec"
                          + (" and is required, so a spec-following caller cannot call this"
                             " tool" if required else ""),
            })
        for pname in sorted(set(sp) & set(lv)):
            s, l = sp[pname], lv[pname]
            if s["type"] and l["type"] and s["type"] != l["type"]:
                findings.append({
                    "kind": "type_mismatch", "tool": name, "param": pname,
                    "detail": f"spec type '{s['type']}' != tool type '{l['type']}'",
                })
            if s["enum"] is not None and l["enum"] is not None and s["enum"] != l["enum"]:
                findings.append({
                    "kind": "enum_mismatch", "tool": name, "param": pname,
                    "detail": f"spec enum {s['enum']} != tool enum {l['enum']}",
                })
            if s["required"] != l["required"]:
                findings.append({
                    "kind": "required_mismatch", "tool": name, "param": pname,
                    "detail": f"spec required={s['required']} != tool required={l['required']}",
                })

    for f in findings:
        f["blocking"] = f["kind"] in BLOCKING_FINDING_KINDS
    blocking = [f for f in findings if f["blocking"]]

    return {
        "passed": len(blocking) == 0,
        "total_tools": len(tools),
        # Raw spec count, not the collision-collapsed index: it would otherwise under-report
        # exactly when an `ambiguous_tool` finding says something is wrong.
        "spec_operations": len(spec_params),
        "findings": findings,
        "blocking_findings": len(blocking),
    }


def skipped_result(reason: str) -> dict[str, Any]:
    """Advisory 'nothing to check' verdict: ``skipped`` frees the gate, ``passed``
    stays False so nothing reads it as a real conformance pass.
    """
    return {
        "passed": False, "skipped": True, "reason": reason,
        "findings": [], "blocking_findings": 0,
    }


# Statuses meaning "no spec is served here", not "the fetch broke".
_SPEC_ABSENT_STATUSES = (404, 405, 501)


def _summarize(findings: list[dict], limit: int = 10) -> str:
    """One-line digest, blocking first so truncation drops noise rather than signal."""
    ordered = sorted(findings, key=lambda f: not f.get("blocking"))
    parts = []
    for f in ordered[:limit]:
        loc = f.get("tool", "")
        if f.get("param"):
            loc = f"{loc}.{f['param']}"
        parts.append(f"{f['kind']}:{loc}")
    extra = len(ordered) - limit
    if extra > 0:
        parts.append(f"(+{extra} more)")
    return "; ".join(parts)


class VerifySpecConformanceTaskStep(TaskStep):
    type: ClassVar[str] = "verify_spec_conformance"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifySpecConformanceTaskStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.env.env import Env

        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")

        env = Env.get(self.env_id, deployed.env_version)
        environment_name = env.environment_name

        # 1. Fetch the baked OpenAPI spec; no spec = skip, not block.
        base_url = legacy_protocol.environment_base_url(deployed.gateway_url, environment_name, mcp=True)
        spec_url = f"{base_url}/openapi.yaml"
        logger.info(f"Fetching OpenAPI spec for '{environment_name}' at {spec_url}")
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            resp = await client.get(spec_url)
            if resp.status_code in _SPEC_ABSENT_STATUSES:
                return self._record(
                    context, env, environment_name,
                    skipped_result(f"no OpenAPI spec served at {spec_url} (HTTP {resp.status_code})"),
                )
            resp.raise_for_status()
            spec = yaml.safe_load(resp.text)
        if not isinstance(spec, dict) or not spec.get("paths"):
            # Served but unusable (error page, empty doc): skip rather than emit a
            # wall of bogus `extra_tool`s.
            return self._record(
                context, env, environment_name,
                skipped_result(f"OpenAPI spec at {spec_url} has no usable 'paths' section"),
            )

        # 2. Read the live tool surface. Same explicit timeout as the spec fetch so a
        #    slow gateway can't trip httpx's short default and block the gate.
        mcp_url = deployed.mcp_url
        logger.info(f"Connecting to MCP at {mcp_url} to read live tools")
        http_client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S)
        try:
            async with streamable_http_client(mcp_url, http_client=http_client) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
        finally:
            await http_client.aclose()
        tools = tools_result.tools

        # 3. Diff. Without the tool-name prefix every tool reads as both missing and extra.
        result = diff_conformance(spec, tools, service_name=environment_name)
        logger.info(
            f"Spec conformance for '{environment_name}': passed={result['passed']} "
            f"tools={result['total_tools']} spec_ops={result['spec_operations']} "
            f"blocking={result['blocking_findings']} findings={len(result['findings'])}"
        )

        # 4. Persist (record-only): a failed verdict doesn't raise, so validate()
        #    completes and cleans up its sandboxes. `put` enforces.
        return self._record(context, env, environment_name, result)

    def _record(self, context: TaskStepContext, env, environment_name: str, result: dict) -> TaskStepContext:
        """Persist a verdict to env + context metadata and log it. ``merge_metadata``
        because the schema verifier writes a sibling key on the same doc concurrently.
        """
        env.merge_metadata({"mcp_spec_conformance": result})
        context.metadata.setdefault("verifications", {})["spec_conformance"] = result
        if result.get("skipped"):
            logger.info(f"Spec conformance skipped for '{environment_name}': {result['reason']}")
        elif result["findings"]:
            # Logged whenever there is drift, not only on a failed verdict: informational-
            # only findings still pass, and keying off `passed` silenced the normal case
            # (every server reports its framework tools as `extra_*`). Advisory either
            # way — this never blocks. Level tracks severity so expected framework-tool
            # drift doesn't warn on every run and train operators to ignore it.
            blocking = result["blocking_findings"]
            log = logger.warning if blocking else logger.info
            log(
                f"Spec conformance drift for '{environment_name}' "
                f"({blocking} blocking of {len(result['findings'])} "
                f"finding(s)): {_summarize(result['findings'])}"
            )
        return context
