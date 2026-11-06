"""What ``agent-env env website new`` writes: a site like the Notes example, from the templates beside this module, and
optionally from an MCP server's tools, one stub ``@api`` route each. A server's tools are read without running it: from
the ``@tool`` and ``@<server>.tool`` methods of its Python source (agentenv_protocol's and FastMCP's alike), or from a
JSON file of them, as ``tools/list`` answers, as a list, or as an environment card's ``capabilities.tools``."""
from __future__ import annotations

import ast
import inspect
import json
import keyword
import re
from importlib import metadata
from pathlib import Path
from typing import Any

import agentenv_protocol.website as website_sdk
from agentenv_protocol.website import SESSION_PATH, WebsiteEnvironment
from pydantic import BaseModel

from agent_env.env.envs.website import domains_problem

TEMPLATE_DIR = Path(__file__).resolve().parent / "website_template"
CLIENT_DIR = Path(website_sdk.__file__).resolve().parent / "client"
NAME = re.compile(r"^[a-z][a-z0-9-]*$")
_TOKEN = re.compile(r"{{(\w+)}}")
_SKIPPED_DIRS = {"node_modules", ".venv", "venv", "__pycache__", ".git", "dist", "build"}
_CONTEXT_TYPES = {"ToolContext", "Context"}
_READ_VERBS = ("get", "list", "search", "find", "read", "fetch", "query", "lookup", "show", "count")
_VERBS = _READ_VERBS + ("create", "add", "new", "update", "edit", "set", "delete", "remove", "send", "post", "put",
                        "patch", "make", "write", "cancel", "book", "submit")
_BUILTIN_PATHS = {"/api/users", "/api/sign-in", SESSION_PATH}
_JSON_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "object": "dict",
               "array": "list", "null": "None"}
_PY_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "dict": "object",
             "list": "array", "Dict": "object", "List": "array", "tuple": "array", "Tuple": "array", "set": "array"}


class ScaffoldError(Exception):
    """Why a site can't be scaffolded as asked."""


def scaffold(target: Path, name: str, *, domains: list[str] | None = None, tools: list[dict] | None = None,
             source: str | None = None, force: bool = False) -> list[Path]:
    """Write the site ``name`` into ``target`` and return the files written. Its card declares ``NAME.test``;
    ``domains``, the names a deploy reaches it by, go in its ``env.toml``, which wins over the card. With ``tools``,
    its routes are their stubs; ``source`` says where they came from. A non-empty ``target`` is refused unless
    ``force``."""
    if not NAME.match(name):
        raise ScaffoldError(f"a site's name is lowercase letters, digits and hyphens, starting with a letter, not {name!r}")
    if problem := domains_problem(list(domains or [])):
        raise ScaffoldError(problem)
    if target.exists() and not target.is_dir():
        raise ScaffoldError(f"{target} is a file")
    if target.is_dir() and any(target.iterdir()) and not force:
        raise ScaffoldError(f"{target} is not empty; pass --force to write into it")
    values = {"name": name, "title": _title(name), "class_name": _class_name(name), "domain": f"{name}.test",
              "domains_line": f"domains = {json.dumps(list(domains))}\n" if domains else "",
              "protocol": _protocol_requirement(), "source": source or "an MCP server",
              "source_note": f" from the tools of {source}" if tools is not None else ""}
    values.update(state=_render("site-state.py.tmpl", {}), part_data=_render("site-parts.py.tmpl", {}))
    files: dict[str, str] = {}
    for out, template in (("data/seed.json", "data/seed.json.tmpl"), ("web/index.html", "web/index.html.tmpl"),
                          ("web/src/style.css", "web/src/style.css.tmpl"), ("web/package.json", "web/package.json.tmpl"),
                          ("web/vite.config.js", "web/vite.config.js.tmpl"), ("Dockerfile", "Dockerfile.tmpl"),
                          ("Dockerfile.dockerignore", "Dockerfile.dockerignore.tmpl"), ("env.toml", "env.toml.tmpl"),
                          ("README.md", "README.md.tmpl"), (".gitignore", "gitignore.tmpl")):
        files[out] = _render(template, values)
    if tools is None:
        files["site.py"] = _render("site.py.tmpl", values)
        files["web/src/main.js"] = _render("web/src/main.js.tmpl", values)
    else:
        if not tools:
            raise ScaffoldError("there are no tools to make routes of")
        stubs = _stubs(tools)
        files["site.py"] = _render("site-tools.py.tmpl", {**values, **_site_parts(stubs)})
        files["data/seed.json"] = json.dumps(seed_for(tools), indent=2) + "\n"
        routes = [{"name": s["name"], "method": s["method"], "path": s["path"].lstrip("/"),
                   "description": s["summary"], "example": s["example"]} for s in stubs]
        files["web/src/main.js"] = _render("web/src/main-tools.js.tmpl",
                                           {**values, "routes_json": json.dumps(routes, indent=2)})
    for client in ("site-client.js", "site-client.d.ts"):
        files[f"web/src/{client}"] = (CLIENT_DIR / client).read_text(encoding="utf-8")
    written = []
    for relative, text in files.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return sorted(written)


def tools_from_mcp(path: Path) -> list[dict]:
    """The tools of the MCP server at ``path``, a folder or a ``.py`` file of its source, or a tools JSON file, each
    as ``tools/list`` lists one: ``{name, description, inputSchema}``."""
    if path.is_file() and path.suffix == ".json":
        return tools_from_json(path)
    if path.is_file():
        sources = [path]
    elif path.is_dir():
        sources = sorted(p for p in path.rglob("*.py") if not _SKIPPED_DIRS & set(p.relative_to(path).parts)
                         and not p.name.startswith("test_") and not p.name.endswith("_test.py"))
    else:
        raise ScaffoldError(f"no MCP server at {path}")
    tools: dict[str, dict] = {}
    for source in sources:
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (SyntaxError, UnicodeDecodeError, OSError) as e:
            raise ScaffoldError(f"can't read {source}: {e}") from e
        for found in _tools_in(tree):
            tools.setdefault(found["name"], found)
    if not tools:
        raise ScaffoldError(f"found no @tool methods in {path}; pass the server's tools as JSON with --from-tools-json")
    return list(tools.values())


def tools_from_json(path: Path) -> list[dict]:
    """The tools a JSON file lists: ``tools/list``'s answer (or its JSON-RPC response), a list of tools, or an
    environment card with ``capabilities.tools``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ScaffoldError(f"can't read tools from {path}: {e}") from e
    if isinstance(data, dict) and isinstance(data.get("result"), dict):
        data = data["result"]
    if isinstance(data, dict) and isinstance((data.get("capabilities") or {}).get("tools"), list):
        data = data["capabilities"]["tools"]
    elif isinstance(data, dict) and isinstance(data.get("tools"), list):
        data = data["tools"]
    if not isinstance(data, list) or not all(isinstance(t, dict) and isinstance(t.get("name"), str) for t in data):
        raise ScaffoldError(f"{path} lists no tools: expected tools/list's {{\"tools\": [...]}}, a list of "
                            "{name, description, inputSchema}, or an environment card")
    return [{"name": t["name"], "description": t.get("description") or "",
             "inputSchema": t.get("inputSchema") or t.get("input_schema") or {"type": "object", "properties": {}}}
            for t in data]


def seed_for(tools: list[dict]) -> dict:
    """A seed for a site made from ``tools``: an acting user, the users to sign in as, and an empty collection for
    what each tool's name is about (``search_emails`` and ``send_email`` both ``emails``)."""
    seed: dict = {"current_user": "user", "users": [{"id": "user", "name": "Sample User"}]}
    for tool in tools:
        collection = _collection(tool["name"])
        if collection and collection not in seed:
            seed[collection] = []
    return seed


def _render(template: str, values: dict[str, str]) -> str:
    text = (TEMPLATE_DIR / template).read_text(encoding="utf-8")

    def value(match: re.Match) -> str:
        if match.group(1) not in values:
            raise KeyError(f"{template} names {match.group(0)}, which nothing fills")
        return values[match.group(1)]

    return _TOKEN.sub(value, text)


def _title(name: str) -> str:
    return " ".join(part.capitalize() for part in name.split("-") if part)


def _class_name(name: str) -> str:
    parts = [part.capitalize() for part in name.split("-") if part]
    return "".join(parts) + ("" if parts[-1] == "Site" else "Site")


def _protocol_requirement() -> str:
    """The protocol package the site's image installs: at least the version this CLI has, which has the website SDK."""
    try:
        return f"agentenv-framework-protocol>={metadata.version('agentenv-framework-protocol')}"
    except metadata.PackageNotFoundError:
        return "agentenv-framework-protocol"


def _words(name: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()) if w]


def _collection(name: str) -> str | None:
    words = _words(name)
    if words and words[0] in _VERBS:
        words = words[1:]
    if not words:
        return None
    if not words[-1].endswith("s"):
        words[-1] += "s"
    return "_".join(words)


def _identifier(name: str, taken: set[str]) -> str:
    """A Python name for ``name``, not a keyword, a BaseModel attribute or one of ``taken``."""
    ident = "_".join(_words(name)) or "value"
    if ident[0].isdigit():
        ident = f"n_{ident}"
    while keyword.iskeyword(ident) or hasattr(BaseModel, ident) or ident in taken or ident in ("self", "ctx"):
        ident += "_"
    taken.add(ident)
    return ident


def _py_type(schema: Any, typing: set[str]) -> str:
    """The Python annotation of a JSON schema, adding the ``typing`` names it uses to ``typing``."""
    if not isinstance(schema, dict):
        typing.add("Any")
        return "Any"
    options = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(options, list):
        kinds = [o for o in options if not (isinstance(o, dict) and o.get("type") == "null")]
        inner = _py_type(kinds[0], typing) if len(kinds) == 1 else _any(typing)
        if len(kinds) < len(options):
            typing.add("Optional")
            return f"Optional[{inner}]"
        return inner
    if isinstance(schema.get("enum"), list) and schema["enum"] and all(isinstance(v, (str, int)) for v in schema["enum"]):
        typing.add("Literal")
        return f"Literal[{', '.join(repr(v) for v in schema['enum'])}]"
    kind = schema.get("type")
    if isinstance(kind, list):
        real = [k for k in kind if k != "null"]
        inner = _py_type({**schema, "type": real[0]}, typing) if len(real) == 1 else _any(typing)
        if "null" in kind:
            typing.add("Optional")
            return f"Optional[{inner}]"
        return inner
    if kind == "array":
        return f"list[{_py_type(schema['items'], typing)}]" if isinstance(schema.get("items"), dict) else "list"
    if kind in _JSON_TYPES and kind != "null":
        return _JSON_TYPES[kind]
    return _any(typing)


def _any(typing: set[str]) -> str:
    typing.add("Any")
    return "Any"


def _scalar(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    options = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(options, list):
        return all(_scalar(o) or (isinstance(o, dict) and o.get("type") == "null") for o in options)
    if isinstance(schema.get("enum"), list):
        return True
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    return bool(kinds) and all(k in ("string", "integer", "number", "boolean", "null") for k in kinds)


def _literal(value: Any) -> str | None:
    """``value`` as Python source, when it is plain JSON."""
    try:
        source = repr(value)
        return source if ast.literal_eval(source) == value else None
    except (ValueError, SyntaxError):
        return None


def _stubs(tools: list[dict]) -> list[dict]:
    """Each tool's route: its function and model names, verb, path, parameters and the example the page sends. A
    function is never named like the site class's own methods or a lowercase name ``site-tools.py.tmpl`` imports,
    such as ``api``, which a method of that name would shadow for the routes declared after it."""
    functions: set[str] = {"api", "ready_check", "website", "environment_card", "reset_data", "add_data", "get_data",
                           "seeded", "reset", "add", "export", "data", "list_users", "sign_in", *dir(WebsiteEnvironment)}
    names: set[str] = {"list_users", "sign_in"}
    models: set[str] = {"SignIn"}
    paths: set[str] = set(_BUILTIN_PATHS)
    stubs = []
    for tool in tools:
        schema = tool.get("inputSchema") if isinstance(tool.get("inputSchema"), dict) else {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = set(schema.get("required") or [])
        words = _words(tool["name"])
        segment = "-".join(words) or "tool"
        path = f"/api/{segment}"
        while path in paths:
            path += "-tool"
        paths.add(path)
        function = _identifier(tool["name"], functions)
        model = "".join(w.capitalize() for w in words) + "Request"
        if model[0].isdigit():
            model = f"N{model}"
        while model in models:
            model += "_"
        models.add(model)
        taken: set[str] = set()
        params = []
        for key, prop in properties.items():
            prop = prop if isinstance(prop, dict) else {}
            params.append({"key": key, "ident": _identifier(key, taken), "schema": prop,
                           "required": key in required, "description": str(prop.get("description") or "")})
        params.sort(key=lambda p: not p["required"])
        reads = bool(words) and words[0] in _READ_VERBS and all(_scalar(p["schema"]) and p["ident"] == p["key"]
                                                               for p in params)
        route_name = tool["name"]
        while route_name in names:
            route_name += "_tool"
        names.add(route_name)
        example = {p["key"]: _example(p["schema"]) for p in params if p["required"]}
        stubs.append({"tool": tool["name"], "name": route_name, "function": function, "model": model, "path": path,
                      "method": "GET" if reads else "POST", "params": params, "example": example,
                      "description": inspect.cleandoc(str(tool.get("description") or "")),
                      "summary": inspect.cleandoc(str(tool.get("description") or "")).split("\n\n")[0]})
    return stubs


def _example(schema: dict) -> Any:
    if "default" in schema:
        return schema["default"]
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return schema["enum"][0]
    kind = schema.get("type")
    kind = next((k for k in kind if k != "null"), None) if isinstance(kind, list) else kind
    return {"string": "", "integer": 0, "number": 0, "boolean": False, "array": [], "object": {}}.get(kind, None)


def _site_parts(stubs: list[dict]) -> dict[str, str]:
    """The ``{{models}}``, ``{{routes}}`` and ``{{typing_import}}`` of ``site-tools.py.tmpl``."""
    typing: set[str] = set()
    models, routes = [], []
    for stub in stubs:
        doc = _docstring(stub["description"] or f"The {stub['tool']} tool.", 8)
        if stub["method"] == "GET":
            args = ["self", "ctx: ApiContext"]
            for p in stub["params"]:
                annotation = _py_type(p["schema"], typing)
                if p["required"]:
                    args.append(f"{p['ident']}: {annotation}")
                else:
                    default = _literal(p["schema"].get("default")) if "default" in p["schema"] else None
                    if default is None or default == "None":
                        if not annotation.startswith("Optional["):
                            typing.add("Optional")
                            annotation = f"Optional[{annotation}]"
                        default = "None"
                    args.append(f"{p['ident']}: {annotation} = {default}")
            notes = [f"``{p['ident']}``: {p['description']}" for p in stub["params"] if p["description"]]
            if notes:
                doc = _docstring((stub["description"] or f"The {stub['tool']} tool.") + "\n\n" + "\n".join(notes), 8)
            signature = ", ".join(args)
        else:
            fields = []
            for p in stub["params"]:
                annotation = _py_type(p["schema"], typing)
                options = []
                if not p["required"]:
                    default = _literal(p["schema"].get("default")) if "default" in p["schema"] else None
                    if default is None or default == "None":
                        if not annotation.startswith("Optional["):
                            typing.add("Optional")
                            annotation = f"Optional[{annotation}]"
                        default = "None"
                    options.append(f"default={default}")
                if p["ident"] != p["key"]:
                    options.append(f"alias={json.dumps(p['key'])}")
                if p["description"]:
                    options.append(f"description={json.dumps(p['description'])}")
                value = f" = Field({', '.join(options)})" if options else ""
                fields.append(f"    {p['ident']}: {annotation}{value}")
            model_doc = _docstring(f"What {stub['tool']} is called with.", 4)
            models.append(f"\n\nclass {stub['model']}(BaseModel):\n{model_doc}\n" + "\n".join(fields))
            signature = f"self, body: {stub['model']}, ctx: ApiContext"
        routes.append(f"\n    @api.{stub['method'].lower()}({json.dumps(stub['path'])}, name={json.dumps(stub['name'])})\n"
                      f"    def {stub['function']}({signature}) -> Any:\n{doc}\n"
                      f"        raise _stub({json.dumps(stub['tool'])})\n")
    typing.update(("Any", "Optional"))
    return {"models": "".join(models) + ("\n" if models else ""), "routes": "".join(routes).rstrip("\n") + "\n",
            "typing_import": f"from typing import {', '.join(sorted(typing))}\n"}


def _docstring(text: str, indent: int) -> str:
    pad = " " * indent
    text = text.replace("\\", "\\\\").replace('"', '\\"').strip()
    lines = text.splitlines() or [""]
    if len(lines) == 1:
        return f'{pad}"""{lines[0]}"""'
    body = "\n".join(f"{pad}{line}" if line else "" for line in lines[1:])
    return f'{pad}"""{lines[0]}\n{body}"""'


def _tools_in(tree: ast.Module) -> list[dict]:
    """The tools ``tree`` declares: each function under a ``tool`` decorator, named by the decorator's ``name`` (or
    its first argument) else the function's name, described by its ``description`` else the docstring."""
    tools = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorator = next((d for d in node.decorator_list if _is_tool(d)), None)
        if decorator is None:
            continue
        name, description = node.name, None
        if isinstance(decorator, ast.Call):
            given = next((k.value for k in decorator.keywords if k.arg == "name"), None)
            if given is None and decorator.args:
                given = decorator.args[0]
            if isinstance(given, ast.Constant) and isinstance(given.value, str):
                name = given.value
            described = next((k.value for k in decorator.keywords if k.arg == "description"), None)
            if isinstance(described, ast.Constant) and isinstance(described.value, str):
                description = described.value
        tools.append({"name": name, "description": description or ast.get_docstring(node) or "",
                      "inputSchema": _input_schema(node.args)})
    return tools


def _is_tool(decorator: ast.expr) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return (isinstance(target, ast.Name) and target.id == "tool") or (
        isinstance(target, ast.Attribute) and target.attr == "tool")


def _input_schema(args: ast.arguments) -> dict:
    """The JSON schema a tool's parameters ask for, read from their annotations, ``Field``\\ s and defaults; ``self``,
    ``cls`` and ``ToolContext``/``Context`` parameters are not the caller's."""
    positional = [*args.posonlyargs, *args.args]
    defaults: dict[str, ast.expr | None] = dict.fromkeys(a.arg for a in [*positional, *args.kwonlyargs])
    for arg, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
        defaults[arg.arg] = default
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        defaults[arg.arg] = default
    properties: dict[str, dict] = {}
    required = []
    for arg in [*positional, *args.kwonlyargs]:
        if arg.arg in ("self", "cls") or _is_context(arg.annotation):
            continue
        schema = _annotation_schema(arg.annotation)
        default = defaults[arg.arg]
        has_default = default is not None
        if isinstance(default, ast.Call) and _call_name(default) == "Field":
            schema.update(_field_options(default))
            has_default = "default" in schema
        elif default is not None:
            try:
                schema["default"] = ast.literal_eval(default)
            except (ValueError, TypeError, SyntaxError):
                pass
        if "_required" in schema:
            has_default = not schema.pop("_required")
        if "default" in schema:
            has_default = True
        properties[arg.arg] = schema
        if not has_default:
            required.append(arg.arg)
    return {"type": "object", "properties": properties, "required": required}


def _is_context(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    names = {n.id for n in ast.walk(annotation) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(annotation) if isinstance(n, ast.Attribute)}
    return bool(names & _CONTEXT_TYPES)


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    return call.func.attr if isinstance(call.func, ast.Attribute) else None


def _field_options(call: ast.Call) -> dict:
    """What a ``Field(...)`` says of its parameter: its description, and its default or that it has none."""
    options: dict = {}
    for k in call.keywords:
        if k.arg == "description" and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str):
            options["description"] = k.value.value
    given = next((k.value for k in call.keywords if k.arg == "default"), call.args[0] if call.args else None)
    if given is not None:
        if isinstance(given, ast.Constant) and given.value is Ellipsis:
            options["_required"] = True
        else:
            try:
                options["default"] = ast.literal_eval(given)
            except (ValueError, TypeError, SyntaxError):
                options["_required"] = False
    elif any(k.arg == "default_factory" for k in call.keywords):
        options["_required"] = False
    return options


def _annotation_schema(node: ast.expr | None) -> dict:
    """The JSON schema of a parameter's annotation, as far as the source shows it."""
    if node is None:
        return {}
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _annotation_schema(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return {}
    if isinstance(node, ast.Constant) and node.value is None:
        return {"type": "null"}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _union([_annotation_schema(node.left), _annotation_schema(node.right)])
    if isinstance(node, (ast.Name, ast.Attribute)):
        name = node.id if isinstance(node, ast.Name) else node.attr
        return {"type": _PY_TYPES[name]} if name in _PY_TYPES else {}
    if isinstance(node, ast.Subscript):
        outer = node.value.id if isinstance(node.value, ast.Name) else getattr(node.value, "attr", "")
        inner = node.slice
        items = list(inner.elts) if isinstance(inner, ast.Tuple) else [inner]
        if outer == "Annotated":
            schema = _annotation_schema(items[0])
            for extra in items[1:]:
                if isinstance(extra, ast.Call) and _call_name(extra) == "Field":
                    schema.update(_field_options(extra))
            return schema
        if outer == "Optional":
            return _union([_annotation_schema(items[0]), {"type": "null"}])
        if outer == "Union":
            return _union([_annotation_schema(i) for i in items])
        if outer == "Literal":
            values = [i.value for i in items if isinstance(i, ast.Constant)]
            return {"enum": values} if values else {}
        if outer in _PY_TYPES and _PY_TYPES[outer] == "array":
            return {"type": "array", "items": _annotation_schema(items[0])}
        if outer in _PY_TYPES:
            return {"type": _PY_TYPES[outer]}
    return {}


def _union(options: list[dict]) -> dict:
    real = [o for o in options if o.get("type") != "null"]
    if len(real) == len(options):
        return {"anyOf": options} if len(options) > 1 else options[0]
    return {"anyOf": [*real, {"type": "null"}]} if real else {"type": "null"}
