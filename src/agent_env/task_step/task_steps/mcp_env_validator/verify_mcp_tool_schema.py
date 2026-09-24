"""Verify MCP tool schemas for a deployed environment."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


def resolve_defs_ref(param_schema: dict, schema: dict) -> dict:
    """Resolve a ``#/$defs/`` ``$ref`` pointer against ``$defs`` in the top-level schema.

    Module-level so it can be imported and reused by other verifiers (e.g.
    ``verify_spec_conformance``) without coupling to a task-step class.
    """
    ref = param_schema.get("$ref", "")
    if not ref.startswith("#/$defs/"):
        return param_schema
    def_name = ref[len("#/$defs/"):]
    resolved = schema.get("$defs", {}).get(def_name)
    if resolved is None:
        return param_schema
    # Fields on the original property (e.g. description) override the $defs entry
    merged = {**resolved, **{k: v for k, v in param_schema.items() if k != "$ref"}}
    return merged


class VerifyMCPToolSchemaTaskStep(TaskStep):
    type: ClassVar[str] = "verify_mcp_tool_schema"
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
    def from_dict(cls, data: dict) -> VerifyMCPToolSchemaTaskStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"])

    @staticmethod
    def _resolve_ref(param_schema: dict, schema: dict) -> dict:
        """Resolve a $ref pointer against $defs in the top-level schema.

        Thin delegator to the module-level :func:`resolve_defs_ref`; kept for
        backward compatibility with existing call sites.
        """
        return resolve_defs_ref(param_schema, schema)

    @staticmethod
    def _check_parameter_issues(tool_name: str, schema: dict) -> list[str]:
        """Check a single tool's inputSchema for parameter quality issues."""
        issues = []
        properties = schema.get("properties", {})
        if not properties:
            return issues

        if "required" in schema and not schema["required"]:
            issues.append(f"tool '{tool_name}' has empty 'required' array")

        for param_name, param_schema in properties.items():
            if not param_schema or not isinstance(param_schema, dict):
                issues.append(f"parameter '{param_name}' has empty or invalid schema")
                continue
            param_schema = VerifyMCPToolSchemaTaskStep._resolve_ref(param_schema, schema)
            has_type = "type" in param_schema
            if not has_type and "anyOf" in param_schema:
                resolved_variants = [VerifyMCPToolSchemaTaskStep._resolve_ref(v, schema) if isinstance(v, dict) else v for v in param_schema["anyOf"]]
                has_type = any("type" in v for v in resolved_variants if isinstance(v, dict))
            if not has_type:
                issues.append(f"parameter '{param_name}' missing explicit type")
            if "description" not in param_schema:
                issues.append(f"parameter '{param_name}' missing description")
            if param_schema.get("type") == "array" and "items" not in param_schema:
                issues.append(f"parameter '{param_name}' is array without items type")
            if param_schema.get("type") == "object" and "properties" not in param_schema:
                issues.append(f"parameter '{param_name}' is object without properties")
            if "enum" in param_schema and not param_schema["enum"]:
                issues.append(f"parameter '{param_name}' has empty enum list")

        return issues

    @staticmethod
    def validate_tool_descriptions(tools: list) -> dict[str, Any]:
        """Check tool descriptions and parameter quality. Includes full tool schemas."""
        missing = [t.name for t in tools if not (t.description or "").strip()]
        tool_schemas = [
            {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema or {}}
            for t in tools
        ]
        parameter_issues: dict[str, list[str]] = {}
        for t in tools:
            issues = VerifyMCPToolSchemaTaskStep._check_parameter_issues(t.name, t.inputSchema or {})
            if issues:
                parameter_issues[t.name] = issues

        passed = len(missing) == 0 and len(parameter_issues) == 0
        return {"passed": passed, "total_tools": len(tools), "tools_missing_description": missing, "parameter_issues": parameter_issues, "tools": tool_schemas}

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.env.env import Env

        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")

        mcp_url = deployed.mcp_url
        logger.info(f"Connecting to MCP at {mcp_url} to verify tool schemas")
        http_client = httpx.AsyncClient()
        try:
            async with streamable_http_client(mcp_url, http_client=http_client) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
        finally:
            await http_client.aclose()

        tools = tools_result.tools
        logger.info(f"Retrieved {len(tools)} tool(s) from {mcp_url}")

        result = self.validate_tool_descriptions(tools)
        logger.info(f"Tool schema validation: passed={result['passed']}, missing_descriptions={result['tools_missing_description']}")

        # merge_metadata: a sibling verifier writes its own key on the same doc.
        env = Env.get(self.env_id, deployed.env_version)
        env.merge_metadata({"mcp_tool_schema_validation": result})

        if "verifications" not in context.metadata:
            context.metadata["verifications"] = {}
        context.metadata["verifications"]["mcp_tool_schema"] = result

        return context
