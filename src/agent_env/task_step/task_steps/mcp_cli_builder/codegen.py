"""Generate a Click-based CLI script from a manifest and live tools.

When an Interface Manifest is available, it supplies entity/action structure
while the gateway's live tool schemas supply invocation details. Servers
without a manifest retain the legacy flat tool CLI. Both modes query
`/step list_tools` on each invocation so role filtering stays live.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from agentenv_protocol import MANIFEST_VERSION, manifest_compatible

from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER

GATEWAY_URL_ENV_VAR = "AGENT_ENV_GATEWAY_URL"
ROLE_ENV_VAR = "AGENT_ENV_ROLE"
DEFAULT_CLI_ROLE = "cli"
_COMMAND_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")
_WORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+")
# Root subcommands the renderer owns; an entity must not normalize to one.
_RESERVED_ENTITY_NAMES = frozenset({"actions"})


def _entity_command_name(entity_name: str, environment_name: str) -> str:
    """The entity's group name, or "" when it is its service's namesake.

    A namesake entity (Apartment in apartment) strips to nothing; returning ""
    tells the caller to render its verbs at the root rather than repeat the token.
    """
    words = _WORD_RE.findall(entity_name)
    service_words = [
        part.lower()
        for part in re.split(r"[-_]+", environment_name)
        if part
    ]
    lowered = [word.lower() for word in words]
    if service_words and lowered[: len(service_words)] == service_words:
        lowered = lowered[len(service_words):]
    return "-".join(lowered)


def _action_command_name(tool_name: str, environment_name: str) -> str:
    prefix = f"{environment_name}_"
    name = tool_name[len(prefix):] if tool_name.startswith(prefix) else tool_name
    return re.sub(r"_+", "-", name).strip("-").lower()


def _validate_params(container: Any, label: str) -> None:
    """Reject a malformed params projection at build time, not CLI-invocation time."""
    if not isinstance(container, dict):
        raise ValueError(f"Manifest {label} must be an object")
    params = container.get("params")
    if params is None:
        return
    if not isinstance(params, list):
        raise ValueError(f"Manifest {label} params must be a list")
    for param in params:
        if not isinstance(param, dict):
            raise ValueError(f"Manifest {label} params entries must be objects")
        name = param.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Manifest {label} has a param with no usable name")


def _prepare_interface_manifest(
    interface_manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if interface_manifest is None:
        return None
    if not isinstance(interface_manifest, dict):
        raise ValueError("interface_manifest must be a JSON object")
    version = interface_manifest.get("manifest_version")
    if not manifest_compatible(version):
        raise ValueError(
            f"Unsupported Interface Manifest version {version!r}; this renderer "
            f"supports {MANIFEST_VERSION!r} (same major, and same minor while 0.x)"
        )
    interface = interface_manifest.get("interface")
    if interface not in (None, "cli"):
        raise ValueError(
            f"Expected a CLI Interface Manifest, got interface {interface!r}"
        )
    environment_name = interface_manifest.get("service")
    if not isinstance(environment_name, str) or not environment_name:
        raise ValueError("Interface Manifest must include a non-empty service")

    prepared = copy.deepcopy(interface_manifest)
    entity_names: set[str] = set()
    root_verbs: set[str] = set()
    rooted_entity: str | None = None
    for entity in prepared.get("entities") or []:
        raw_name = entity.get("entity") or ""
        if not _WORD_RE.findall(raw_name):
            raise ValueError(f"Entity {raw_name!r} has no usable CLI command name")

        # In the CLI projection, commands are keyed by verb (list/get/...); the
        # key is the subcommand name, so there is nothing further to assign.
        commands = entity.get("commands")
        if commands is not None and not isinstance(commands, dict):
            raise ValueError(f"Entity {raw_name!r} commands must be an object")
        for verb, command in (commands or {}).items():
            _validate_params(command, f"entity {raw_name!r} command {verb!r}")

        command_name = _entity_command_name(raw_name, environment_name)
        if not command_name:
            # Namesake entity: `apartment list`, not `apartment apartment list`.
            if rooted_entity is not None:
                raise ValueError(
                    f"Entities {rooted_entity!r} and {raw_name!r} both match the "
                    f"service name; only one can supply root commands"
                )
            rooted_entity = raw_name
            root_verbs = set(commands or {})
            entity["_command_name"] = None
            continue
        if command_name in _RESERVED_ENTITY_NAMES:
            raise ValueError(
                f"Entity CLI command {command_name!r} collides with a reserved "
                f"root command; rename the entity or its service prefix"
            )
        if command_name in entity_names:
            raise ValueError(f"Duplicate entity CLI command {command_name!r}")
        entity_names.add(command_name)
        entity["_command_name"] = command_name

    collisions = sorted(root_verbs & (entity_names | _RESERVED_ENTITY_NAMES))
    if collisions:
        raise ValueError(
            f"Root command {collisions[0]!r} from entity {rooted_entity!r} collides "
            f"with another root command"
        )

    action_names: set[str] = set()
    for action in prepared.get("actions") or []:
        command_name = _action_command_name(action.get("tool") or "", environment_name)
        if not command_name:
            raise ValueError(
                f"Action {action.get('name') or action.get('tool')!r} has no usable "
                f"CLI command name"
            )
        if command_name in action_names:
            raise ValueError(f"Duplicate action CLI command {command_name!r}")
        action_names.add(command_name)
        action["_command_name"] = command_name
        _validate_params(action, f"action {command_name!r}")
    return prepared


def generate_cli_script(
    command_name: str,
    interface_manifest: dict[str, Any] | None = None,
    environment_name: str | None = None,
) -> str:
    if not _COMMAND_NAME_RE.match(command_name):
        raise ValueError(f"command_name {command_name!r} must match {_COMMAND_NAME_RE.pattern!r}")
    prepared_manifest = _prepare_interface_manifest(interface_manifest)
    return _SCRIPT.format(
        command_name=command_name,
        environment_name=repr(environment_name or command_name),
        interface_manifest_literal=repr(prepared_manifest),
        gateway_env_var=GATEWAY_URL_ENV_VAR,
        role_env_var=ROLE_ENV_VAR,
        default_role=DEFAULT_CLI_ROLE,
        role_header=AGENT_ENV_ROLE_HEADER,
    )


_SCRIPT = '''#!/usr/bin/env python3
"""Auto-generated CLI for MCP env {command_name}. Discovers tools at runtime."""

import functools
import json
import keyword
import os
import re
import sys

import click
import httpx

GATEWAY_URL_ENV = "{gateway_env_var}"
ROLE_ENV = "{role_env_var}"
DEFAULT_ROLE = "{default_role}"
ROLE_HEADER = "{role_header}"
SERVICE_NAME = {environment_name}
INTERFACE_MANIFEST = {interface_manifest_literal}
_CONFIG_FILE_NAME = ".env"


def _config_file_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), _CONFIG_FILE_NAME)


def _load_sibling_env():
    """Populate os.environ from <bin_dir>/.env (KEY=VALUE per line). Shell env wins."""
    sibling = _config_file_path()
    if not os.path.isfile(sibling):
        return
    for line in open(sibling):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_sibling_env()


def _gateway_url():
    url = os.environ.get(GATEWAY_URL_ENV)
    if not url:
        click.echo(f"error: set {{GATEWAY_URL_ENV}} or write KEY=VALUE pairs to {{_config_file_path()}}", err=True)
        sys.exit(2)
    return url.rstrip("/")


def _role():
    return os.environ.get(ROLE_ENV, DEFAULT_ROLE)


def _post_step(payload):
    response = httpx.post(
        f"{{_gateway_url()}}/step",
        json=payload,
        headers={{ROLE_HEADER: _role()}},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


@functools.lru_cache(maxsize=1)
def _list_tools():
    """Tools for SERVICE_NAME that the current role can access.

    Combines /step list_tools (role-filtered, flat) with /state (server-grouped)
    to scope the listing to this service's tools only.
    """
    role_filtered = _post_step({{"action": "list_tools"}})
    allowed_names = {{t.get("name") for t in (role_filtered.get("tools") or []) if t.get("name")}}
    state_resp = httpx.get(
        f"{{_gateway_url()}}/state",
        headers={{ROLE_HEADER: _role()}},
        timeout=120,
    )
    state_resp.raise_for_status()
    for server in (state_resp.json().get("mcp_servers") or []):
        if server.get("name") == SERVICE_NAME:
            return tuple(t for t in (server.get("tools") or []) if t.get("name") in allowed_names)
    return ()


def _decode_json_flag(raw, schema_type):
    if raw is None:
        return None
    if schema_type in ("array", "object", None):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            click.echo(f"error: expected JSON for value, got {{raw!r}}", err=True)
            sys.exit(2)
    return raw


def _call_tool(tool_name, arguments):
    body = _post_step({{"action": "call_tool", "tool_name": tool_name, "arguments": arguments}})
    if "error" in body:
        click.echo(f"error: {{body['error']}}", err=True)
        sys.exit(1)
    if body.get("isError"):
        text_blocks = [b.get("text", "") for b in body.get("content", []) if b.get("type") == "text"]
        click.echo(f"error: {{''.join(text_blocks) or 'tool returned isError=True'}}", err=True)
        sys.exit(1)
    text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
    try:
        click.echo(json.dumps(json.loads(text), indent=2))
    except json.JSONDecodeError:
        click.echo(text)


def _subcommand_name(tool_name):
    return re.sub(r"_+", "-", tool_name).strip("-")


def _python_identifier(name):
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{{sanitized}}"
    if not sanitized:
        sanitized = "_arg"
    if keyword.iskeyword(sanitized):
        sanitized = f"{{sanitized}}_"
    return sanitized


def _normalize_schema(param_schema):
    schema = param_schema
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        if len(non_null) == 1:
            schema = {{**schema, "type": non_null[0]}}
            schema_type = non_null[0]
        else:
            schema_type = None
    any_of = schema.get("anyOf") or schema.get("oneOf")
    if any_of and schema_type is None:
        non_null = [s for s in any_of if s.get("type") != "null"]
        if len(non_null) == 1:
            inner = non_null[0]
            schema = {{**inner, "description": schema.get("description") or inner.get("description"), "default": schema.get("default", inner.get("default"))}}
            schema_type = inner.get("type")
    return schema_type, schema


def _click_type_for_schema(param_schema):
    """Return (effective_json_type, click_type_obj)."""
    schema_type, normalized = _normalize_schema(param_schema)
    if schema_type == "string":
        enum_values = normalized.get("enum") or []
        if enum_values:
            return "string", click.Choice([str(v) for v in enum_values])
        return "string", str
    if schema_type == "integer":
        return "integer", int
    if schema_type == "number":
        return "number", float
    if schema_type == "boolean":
        return "boolean", bool
    return schema_type, str


def _format_help(text, schema_type):
    base = (text or "").strip().replace("\\n", " ")
    if schema_type in ("array", "object", None):
        suffix = " (JSON-encoded)"
        return (base + suffix) if base else suffix.lstrip()
    return base


def _build_command(
    tool,
    command_name=None,
    parameter_order=None,
    parameter_overrides=None,
    required_override=None,
    help_override=None,
):
    tool_name = tool["name"]
    description = (help_override or tool.get("description") or "").strip().replace("\\n", " ")
    schema = tool.get("parameters") or {{}}
    properties = schema.get("properties") or {{}}
    required_set = set(
        schema.get("required") or []
        if required_override is None
        else required_override
    )
    parameter_overrides = parameter_overrides or {{}}

    options = []
    arg_meta = []
    seen_py_names = set()

    ordered_names = []
    for original_name in (parameter_order or []):
        if original_name in properties and original_name not in ordered_names:
            ordered_names.append(original_name)
    ordered_names.extend(name for name in properties if name not in ordered_names)

    # Both sides derive from the same OpenAPI spec, so drift means the repos are
    # out of sync. Warn rather than fail: the call still uses the live schema.
    if required_override is not None:
        for name in (parameter_order or []):
            if name not in properties:
                click.echo(
                    f"warning: {{tool_name}}: manifest param {{name!r}} is absent from the "
                    f"live tool schema and was dropped",
                    err=True,
                )
        for name in (schema.get("required") or []):
            if name not in required_set:
                click.echo(
                    f"warning: {{tool_name}}: live tool requires {{name!r}} but the manifest "
                    f"does not; rendering it as optional",
                    err=True,
                )

    for original_name in ordered_names:
        param_schema = {{
            **properties[original_name],
            **parameter_overrides.get(original_name, {{}}),
        }}
        flag = "--" + re.sub(r"_+", "-", original_name).strip("-").lower()
        py_name = _python_identifier(original_name)
        candidate, idx = py_name, 1
        while candidate in seen_py_names:
            idx += 1
            candidate = f"{{py_name}}{{idx}}"
        py_name = candidate
        seen_py_names.add(py_name)
        required = original_name in required_set

        effective_type, click_type = _click_type_for_schema(param_schema)
        help_text = _format_help(param_schema.get("description"), effective_type)

        if effective_type == "boolean":
            if "default" in param_schema:
                opt = click.Option(
                    [f"{{flag}}/--no-{{flag.lstrip('-')}}", py_name],
                    default=bool(param_schema["default"]),
                    help=help_text,
                )
            else:
                opt = click.Option(
                    [flag, py_name],
                    is_flag=True,
                    default=False,
                    help=help_text,
                )
        else:
            opt = click.Option(
                [flag, py_name],
                type=click_type,
                required=required,
                default=param_schema.get("default") if not required else None,
                help=help_text,
            )
        options.append(opt)
        arg_meta.append((py_name, original_name, effective_type))

    def _callback(**kwargs):
        arguments = {{}}
        for py_name, original_name, effective_type in arg_meta:
            value = kwargs.get(py_name)
            if effective_type == "boolean":
                arguments[original_name] = value
            elif value is not None:
                if effective_type in ("array", "object", None):
                    arguments[original_name] = _decode_json_flag(value, effective_type)
                else:
                    arguments[original_name] = value
        _call_tool(tool_name, arguments)

    return click.Command(
        name=command_name or _subcommand_name(tool_name),
        params=options,
        callback=_callback,
        help=description,
    )


def _command_rendering(command):
    # The CLI projection already carries each command's final params (required +
    # optional, enums included); the live tool schema fills in the rest.
    params = command.get("params") or []
    ordered = [p.get("name") for p in params if p.get("name")]
    required = [p.get("name") for p in params if p.get("required") and p.get("name")]
    overrides = {{}}
    for p in params:
        name = p.get("name")
        if not name:
            continue
        override = {{key: p[key] for key in ("type", "enum") if p.get(key) is not None}}
        if override:
            overrides[name] = override
    return ordered, overrides, required


def _build_entity_group(entity, tools_by_name):
    commands = {{}}
    for verb, command in (entity.get("commands") or {{}}).items():
        tool = tools_by_name.get(command.get("tool"))
        if tool is None:
            continue
        ordered, overrides, required = _command_rendering(command)
        commands[verb] = _build_command(
            tool,
            command_name=verb,
            parameter_order=ordered,
            parameter_overrides=overrides,
            required_override=required,
            help_override=command.get("summary") or command.get("description"),
        )
    return click.Group(
        name=entity["_command_name"],
        commands=commands,
        help=f"Commands for {{entity.get('entity')}}.",
    )


def _build_actions_group(actions, tools_by_name):
    commands = {{}}
    for action in actions:
        tool = tools_by_name.get(action.get("tool"))
        if tool is None:
            continue
        name = action["_command_name"]
        ordered, overrides, required = _command_rendering(action)
        commands[name] = _build_command(
            tool,
            command_name=name,
            parameter_order=ordered,
            parameter_overrides=overrides,
            required_override=required,
            help_override=action.get("description"),
        )
    return click.Group(
        name="actions",
        commands=commands,
        help="Non-CRUD service actions.",
    )


class _DynamicGroup(click.Group):
    def list_commands(self, ctx):
        try:
            tools = _list_tools()
        except Exception as e:
            click.echo(f"warning: could not fetch tools from gateway: {{e}}", err=True)
            return []
        names = []
        seen = set()
        for t in tools:
            tname = t.get("name")
            if not tname:
                continue
            sub = _subcommand_name(tname)
            if sub in seen:
                continue
            seen.add(sub)
            names.append(sub)
        return sorted(names)

    def get_command(self, ctx, name):
        try:
            tools = _list_tools()
        except Exception as e:
            click.echo(f"error: could not fetch tools from gateway: {{e}}", err=True)
            return None
        for t in tools:
            tname = t.get("name")
            if tname and _subcommand_name(tname) == name:
                return _build_command(t)
        return None


class _ManifestGroup(click.Group):
    def _tools_by_name(self):
        return {{tool["name"]: tool for tool in _list_tools() if tool.get("name")}}

    def list_commands(self, ctx):
        try:
            tools_by_name = self._tools_by_name()
        except Exception as e:
            click.echo(f"warning: could not fetch tools from gateway: {{e}}", err=True)
            return []
        names = []
        for entity in (INTERFACE_MANIFEST.get("entities") or []):
            commands = entity.get("commands") or {{}}
            live = {{
                verb: cmd
                for verb, cmd in commands.items()
                if cmd.get("tool") in tools_by_name
            }}
            if not live:
                continue
            # A namesake entity has no group name; its verbs sit at the root.
            if entity.get("_command_name") is None:
                names.extend(live)
            else:
                names.append(entity["_command_name"])
        if any(
            action.get("tool") in tools_by_name
            for action in (INTERFACE_MANIFEST.get("actions") or [])
        ):
            names.append("actions")
        return sorted(names)

    def get_command(self, ctx, name):
        try:
            tools_by_name = self._tools_by_name()
        except Exception as e:
            click.echo(f"error: could not fetch tools from gateway: {{e}}", err=True)
            return None
        if name == "actions":
            return _build_actions_group(
                INTERFACE_MANIFEST.get("actions") or [],
                tools_by_name,
            )
        for entity in INTERFACE_MANIFEST.get("entities") or []:
            if entity.get("_command_name") is None:
                command = (entity.get("commands") or {{}}).get(name)
                tool = tools_by_name.get((command or {{}}).get("tool"))
                if tool is not None:
                    ordered, overrides, required = _command_rendering(command)
                    return _build_command(
                        tool,
                        command_name=name,
                        parameter_order=ordered,
                        parameter_overrides=overrides,
                        required_override=required,
                        help_override=command.get("summary") or command.get("description"),
                    )
            elif entity.get("_command_name") == name:
                return _build_entity_group(entity, tools_by_name)
        return None


_ROOT_GROUP = _ManifestGroup if INTERFACE_MANIFEST is not None else _DynamicGroup


@click.command(cls=_ROOT_GROUP, help="Auto-generated CLI for {command_name}")
def cli():
    pass


if __name__ == "__main__":
    cli()
'''
