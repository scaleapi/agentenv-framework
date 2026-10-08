"""Fail a change that breaks the plugin surface, unless its title marks the break with ``!``.

The surface is ``BASES`` and ``USED``, the list in README.md "Plugin compatibility"; a unit test
holds the two together. Both trees are read statically with griffe, which reports what breaks a
caller, and the subclass rules below add what breaks a plugin's subclass of a base. A title in the
form ``type(scope)!: summary`` marks the break deliberate: the breaks are printed and the check passes.
"""

from __future__ import annotations

import argparse
import ast
import io
import operator
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from collections.abc import Iterator
from pathlib import Path

import griffe

BASES = (
    "agent_env.env.env.Env",
    "agent_env.task_step.task_step.TaskStep",
    "agent_env.artifact.artifact.Artifact",
    "agent_env.providers.sandbox_providers.sandbox_provider.SandboxProvider",
    "agent_env.providers.env_state.env_state_provider.EnvStateProvider",
    "agent_env.providers.env_providers.env_provider.EnvironmentProvider",
    "agent_env.explorer.plugin.ExplorerPlugin",
)
USED = (
    "agent_env.task_step.context.TaskStepContext", "agent_env.plugins",
    "agent_env.providers.env_providers.env_provider.build_env_provider", "agent_env.env.store.register_env_instance",
    # What a plugin reads of recorded runs.
    "agent_env.task.store.task_instances", "agent_env.task.store.count_task_instances",
    "agent_env.task.store.find_task_instance", "agent_env.task.store.TaskInstance",
    # What an environment provider reads to deploy a built-in env.
    "agent_env.env.envs.mcp_server.MCPServerEnv.docker_image_artifact", "agent_env.env.envs.mcp_server.MCPServerEnv.environment_name",
    "agent_env.env.envs.website.WebsiteEnv.backend_docker_image_artifact",
    "agent_env.env.envs.website.WebsiteEnv.frontend_docker_image_artifact", "agent_env.env.envs.website.WebsiteEnv.environment_name",
    "agent_env.env.envs.multi_env.MultiEnv.mcp_server_envs", "agent_env.env.envs.multi_env.MultiEnv.website_envs",
    "agent_env.env.envs.multi_env.MultiEnv.name", "agent_env.artifact.artifacts.docker_image.DockerImageArtifact.image_name",
)
SURFACE = BASES + USED

ROOT = Path(__file__).resolve().parents[2]
TYPES = ("feat", "fix", "refactor", "docs", "test", "build", "ci", "chore", "perf", "security")
MARKED = re.compile(rf"(?:{'|'.join(TYPES)})(\([^)]+\))?!: ", re.IGNORECASE)
MISPLACED = re.compile(rf"({'|'.join(TYPES)})!(\([^)]+\)): ", re.IGNORECASE)
SHAPES = frozenset({"async", "classmethod", "staticmethod", "property"})
CONSTRUCTORS = frozenset({"__init__", "__init_subclass__", "__post_init__"})
VARIADIC = frozenset({griffe.ParameterKind.var_positional, griffe.ParameterKind.var_keyword})
REQUIREMENT_CHECKS = {"unimplemented": 1, "typed_validator": 0}
_UNREADABLE = object()


def load(src: Path) -> griffe.Module:
    return griffe.load("agent_env", search_paths=[str(src)], resolve_aliases=True, resolve_external=False,
                       allow_inspection=False)


def lookup(pkg: griffe.Module, path: str) -> griffe.Object | None:
    try:
        obj = _target(pkg[path.removeprefix("agent_env.")])
    except KeyError:
        return None
    return _class(obj) or obj


def _target(member: griffe.Object | griffe.Alias) -> griffe.Object | None:
    try:
        return member.final_target if member.is_alias else member
    except (griffe.AliasResolutionError, griffe.CyclicAliasError):
        return None


def _modules(module: griffe.Module) -> Iterator[griffe.Module]:
    yield module
    for sub in module.modules.values():
        if not sub.is_alias:
            yield from _modules(sub)


def _literal(scope: griffe.Object, value: object) -> object:
    """``value`` as a Python value, when it is built from literals, arithmetic, containers and
    module-level names that are: so ``3600 * 2`` and ``7200`` are equal, and a constant behind a
    name is compared by what it holds."""
    try:
        return _evaluate(ast.parse(str(value), mode="eval").body, scope, 0)
    except (SyntaxError, ValueError, TypeError, ArithmeticError, RecursionError):
        return _UNREADABLE


_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
              ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.BitOr: operator.or_}
_BUILDERS = {"frozenset": frozenset, "set": set, "tuple": tuple, "list": list, "dict": dict, "sorted": sorted}


def _evaluate(node: ast.expr, scope: griffe.Object, depth: int) -> object:
    if depth > 8:
        raise ValueError("too deep")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items = []
        for item in node.elts:
            if isinstance(item, ast.Starred):
                items.extend(_evaluate(item.value, scope, depth + 1))
            else:
                items.append(_evaluate(item, scope, depth + 1))
        return {ast.Tuple: tuple, ast.List: list, ast.Set: set}[type(node)](items)
    if isinstance(node, ast.Dict) and None not in node.keys:
        return {_evaluate(k, scope, depth + 1): _evaluate(v, scope, depth + 1) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
        operand = _evaluate(node.operand, scope, depth + 1)
        return -operand if isinstance(node.op, ast.USub) else +operand if isinstance(node.op, ast.UAdd) else not operand
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left, scope, depth + 1), _evaluate(node.right, scope, depth + 1))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _BUILDERS
            and not node.keywords and len(node.args) <= 1):
        return _BUILDERS[node.func.id](*(_evaluate(a, scope, depth + 1) for a in node.args))
    if isinstance(node, (ast.Name, ast.Attribute)):
        member = _object(node, scope)
        if member is not None and member.is_attribute and member.value is not None:
            return _evaluate(ast.parse(str(member.value), mode="eval").body, member, depth + 1)
    raise ValueError(f"not a constant: {ast.dump(node)}")


def _object(node: ast.expr, scope: griffe.Object) -> griffe.Object | None:
    """The object a name or a dotted name such as ``_names.ENVS`` refers to from ``scope``."""
    if isinstance(node, ast.Name):
        return _name(scope, node.id)
    if isinstance(node, ast.Attribute):
        owner = _object(node.value, scope)
        if owner is not None and node.attr in owner.members:
            return _target(owner.members[node.attr])
    return None


def _name(scope: griffe.Object, name: str) -> griffe.Object | None:
    """What ``name`` means in the module ``scope`` belongs to."""
    module = scope.module
    if name not in module.members:
        return None
    return _target(module.members[name])


def _class(obj: griffe.Object | None, depth: int = 0) -> griffe.Class | None:
    """``obj`` as a class, following an assignment such as ``Env = BaseEnv``."""
    if obj is None or depth > 8:
        return None
    if obj.is_class:
        return obj
    if obj.is_attribute and isinstance(obj.value, griffe.ExprName):
        try:
            return _class(_target(obj.modules_collection.get_member(obj.value.canonical_path)), depth + 1)
        except (KeyError, ValueError):
            return None
    return None


def declared_required(pkg: griffe.Module) -> tuple[dict[str, set[str]], list[str]]:
    """What each base's subclasses must implement beyond its abstract methods, by the base's path:
    the ``required`` of each ``unimplemented(cls, base, required)`` and ``typed_validator(base,
    builtins, required)`` call in the package, whose base is at the index ``REQUIREMENT_CHECKS`` gives.
    Also the calls that could not be read, which the caller reports rather than guesses at."""
    found: dict[str, set[str]] = {}
    unreadable: list[str] = []
    for module in _modules(pkg):
        text = module.filepath.read_text() if isinstance(module.filepath, Path) else ""
        if not any(f"{name}(" in text for name in REQUIREMENT_CHECKS):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            unreadable.append(f"{module.path}: cannot parse it ({e.msg}, line {e.lineno})")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if name not in REQUIREMENT_CHECKS:
                continue
            index = REQUIREMENT_CHECKS[name]
            base = node.args[index] if len(node.args) > index else next(
                (k.value for k in node.keywords if k.arg == "base"), None)
            keyword = next((k.value for k in node.keywords if k.arg == "required"), None)
            names = node.args[2] if len(node.args) > 2 else keyword
            if names is None:
                continue
            if isinstance(base, ast.Name) and base.id not in module.members:
                continue  # a helper passing its own parameters through, as typed_validator does
            cls = _class(_name(module, base.id)) if isinstance(base, ast.Name) else None
            value = _literal(module, ast.unparse(names))
            if cls is None or value is _UNREADABLE or isinstance(value, (str, bytes)):
                unreadable.append(f"{module.path}: cannot read the {name}() call on line {node.lineno}, so what "
                                  "its subclasses must implement is unknown")
                continue
            found.setdefault(cls.path, set()).update(value)
    return found, unreadable


def required(cls: griffe.Class, declared: dict[str, set[str]]) -> set[str]:
    abstract = {name for name, member in cls.all_members.items()
                if (target := _target(member)) is not None and "abstractmethod" in target.labels}
    return abstract | declared.get(cls.path, set())


def _shape(obj: griffe.Object) -> str:
    return " ".join([*sorted(obj.labels & SHAPES), obj.kind.value])


def _new_parameters(old: griffe.Function, new: griffe.Function) -> list[griffe.Parameter]:
    """Optional parameters ``new`` adds that an override with ``old``'s signature would not accept."""
    kinds = {p.kind for p in old.parameters}
    names = {p.name for p in old.parameters}
    keywords = griffe.ParameterKind.var_keyword in kinds
    positions = griffe.ParameterKind.var_positional in kinds
    swallowed = {
        griffe.ParameterKind.keyword_only: keywords,
        griffe.ParameterKind.positional_only: positions,
        griffe.ParameterKind.positional_or_keyword: keywords and positions,
    }
    return [p for p in new.parameters
            if p.name not in names and not p.required and p.kind not in VARIADIC and not swallowed[p.kind]]


def _loosened(old: griffe.Function, new: griffe.Function) -> list[str]:
    """What ``new`` accepts that an override with ``old``'s signature would not."""
    kinds = {p.kind for p in old.parameters}
    keywords = griffe.ParameterKind.var_keyword in kinds
    positions = griffe.ParameterKind.var_positional in kinds
    found = []
    for kind, label in ((griffe.ParameterKind.var_keyword, "**"), (griffe.ParameterKind.var_positional, "*")):
        if kind not in kinds and any(p.kind is kind for p in new.parameters):
            found.append(f"now takes {label}{next(p.name for p in new.parameters if p.kind is kind)}")
    olds = {p.name: p for p in old.parameters}
    for p in new.parameters:
        before = olds.get(p.name)
        if before is None or p.kind in VARIADIC:
            continue
        if before.required and not p.required:
            found.append(f"({p.name}): no longer required, so core may leave it out")
        by_position = p.kind is not griffe.ParameterKind.keyword_only and before.kind is griffe.ParameterKind.keyword_only
        by_keyword = p.kind is not griffe.ParameterKind.positional_only and before.kind is griffe.ParameterKind.positional_only
        if (by_position and not positions) or (by_keyword and not keywords):
            found.append(f"({p.name}): can now be passed {'by position' if by_position else 'by keyword'}")
    return found


def subclass_breaks(path: str, old: griffe.Class, new: griffe.Class, old_required: set[str],
                    new_required: set[str]) -> tuple[list[str], set[str]]:
    """What breaks a subclass written against ``old``, and the members whose shape changed."""
    found = []
    for name in sorted(new_required - old_required):
        found.append(f"{path}.{name}: every subclass must now implement it"
                     + ("" if name in old.all_members else " (new)"))
    reshaped = set()
    new_members = new.all_members
    for name, member in old.all_members.items():
        if name not in new_members or not member.is_public:
            continue
        before, after = _target(member), _target(new_members[name])
        if before is None or after is None:
            continue
        if _shape(before) != _shape(after):
            found.append(f"{path}.{name}: {_shape(before)} -> {_shape(after)}")
            reshaped.add(after.path)
        elif before.is_function and name not in CONSTRUCTORS:
            found += [f"{path}.{name}({p.name}): new parameter; an override with the old signature "
                      "does not accept it" for p in _new_parameters(before, after)]
            for change in _loosened(before, after):
                where = f"{path}.{name}" + ("" if change.startswith("(") else ": ")
                found.append(f"{where}{change}; an override with the old signature does not accept that")
    return found, reshaped


def _undocumented(value: object) -> object:
    """``value`` without the ``description=`` of a call, such as a pydantic ``Field``'s."""
    if isinstance(value, griffe.ExprCall):
        kept = [a for a in value.arguments if not (isinstance(a, griffe.ExprKeyword) and a.name == "description")]
        return str(value.function), [str(a) for a in kept]
    return str(value)


def _member(root: griffe.Object, path: str) -> griffe.Object:
    """The object at ``path`` in ``root``'s tree, or ``root`` when the tree has none."""
    try:
        return _target(root.modules_collection.get_member(path)) or root
    except (KeyError, ValueError):
        return root


def _same_value(old_scope: griffe.Object, old: object, new_scope: griffe.Object, new: object) -> bool:
    before = _literal(old_scope, old)
    return before is not _UNREADABLE and before == _literal(new_scope, new)


def _benign(breakage: griffe.Breakage, old_root: griffe.Object, new_root: griffe.Object) -> bool:
    """griffe reports that are not breaks for a plugin. Values are compared by what they evaluate
    to, each in its own tree; type annotations are not compared at all."""
    kind = breakage.kind
    old_scope, new_scope = _member(old_root, breakage.obj.path), _member(new_root, breakage.obj.path)
    if kind is griffe.BreakageKind.ATTRIBUTE_CHANGED_VALUE:
        if breakage.new_value == "unset":
            return False
        labels = breakage.obj.labels
        return (breakage.old_value is None or ("instance-attribute" in labels and "class-attribute" not in labels)
                or _undocumented(breakage.old_value) == _undocumented(breakage.new_value)
                or _same_value(old_scope, breakage.old_value, new_scope, breakage.new_value))
    if kind is griffe.BreakageKind.PARAMETER_MOVED:
        parent = breakage.obj.parent
        return (breakage.obj.name == "__init__" and parent is not None and "dataclass" in parent.labels
                and not breakage.old_value.required)
    if kind is griffe.BreakageKind.PARAMETER_CHANGED_DEFAULT:
        return _same_value(old_scope, breakage.old_value.default, new_scope, breakage.new_value.default)
    return kind is griffe.BreakageKind.RETURN_CHANGED_TYPE


def _describe(breakage: griffe.Breakage, public: dict[str, str]) -> str:
    where = breakage.obj.path
    for target, name in public.items():
        if where == target or where.startswith(f"{target}."):
            where = name + where[len(target):]
            break
    parameter = next((v for v in (breakage.old_value, breakage.new_value) if isinstance(v, griffe.Parameter)), None)
    if parameter is not None and breakage.kind is not griffe.BreakageKind.OBJECT_REMOVED:
        where += f"({parameter.name})"
    old, new = _change(breakage)
    change = f"{old} -> {new}" if old and new else old or new
    return f"{where}: {breakage.kind.value}" + (f": {change}" if change else "")


_UNSHOWN = frozenset({
    griffe.BreakageKind.PARAMETER_MOVED, griffe.BreakageKind.PARAMETER_REMOVED,
    griffe.BreakageKind.PARAMETER_CHANGED_REQUIRED, griffe.BreakageKind.PARAMETER_ADDED_REQUIRED,
    griffe.BreakageKind.OBJECT_REMOVED,
})


def _change(breakage: griffe.Breakage) -> tuple[str, str]:
    """The old and new value a breakage is about, as text; empty where its kind says it all."""
    kind, old, new = breakage.kind, breakage.old_value, breakage.new_value
    if kind in _UNSHOWN:
        return "", ""
    if kind is griffe.BreakageKind.PARAMETER_CHANGED_KIND:
        return str(old.kind.value), str(new.kind.value)
    if kind is griffe.BreakageKind.PARAMETER_CHANGED_DEFAULT:
        return str(old.default), str(new.default)
    if kind is griffe.BreakageKind.OBJECT_CHANGED_KIND:
        return str(old.value), str(new.value)
    if kind is griffe.BreakageKind.CLASS_REMOVED_BASE:
        before, after = ("[" + ", ".join(getattr(b, "canonical_path", str(b)) for b in bases) + "]" for bases in (old, new))
        return before, after
    return str(old), str(new)


def _direct(cls: griffe.Class) -> set[str]:
    return {getattr(b, "canonical_path", str(b)) for b in cls.bases}


def _ancestors(cls: griffe.Class, seen: set[str] | None = None) -> set[str]:
    seen = set() if seen is None else seen
    for path in _direct(cls) - seen:
        seen.add(path)
        try:
            base = _target(cls.modules_collection.get_member(path))
        except (KeyError, ValueError):
            base = None
        if base is not None and base.is_class:
            _ancestors(base, seen)
    return seen


def _exports(module: griffe.Module) -> list[str] | None:
    """The module's ``__all__``, evaluated from its source when griffe could not read it; None
    when neither can."""
    if module.exports:
        return list(module.exports)
    try:
        tree = ast.parse(Path(module.filepath).read_text())
    except (OSError, SyntaxError, TypeError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            try:
                value = _evaluate(node.value, module, 0)
            except (ValueError, TypeError, ArithmeticError, RecursionError):
                return None
            return [str(name) for name in value] if isinstance(value, (list, tuple, set)) else None
    return [] if module.exports is not None else None


def caller_breaks(path: str, old: griffe.Object, new: griffe.Object, reshaped: set[str]) -> list[str]:
    found = []
    if old.is_class:
        kept = _ancestors(new)
        found += [f"{path}: no longer subclasses {base}" for base in sorted(_direct(old) - kept)]
    public = {}
    if old.is_module:
        for module in (old, new):
            for name, member in module.members.items():
                if member.is_alias and member.is_public and (target := _target(member)) is not None:
                    public[target.path] = f"{path}.{name}"
        before, after = _exports(old), _exports(new)
        if before and after is None:
            found.append(f"{path}.__all__: cannot read it, so what the package exports is unknown")
        else:
            found += [f"{path}.{name}: no longer in __all__"
                      for name in sorted(set(before or ()) - set(after or ())) if name in new.members]
    for breakage in griffe.find_breaking_changes(old, new):
        if _benign(breakage, old, new) or breakage.obj.path in reshaped:
            continue
        found.append(_describe(breakage, public))
    return found


def _values(obj: griffe.Object) -> dict[str, tuple[griffe.Object, object]]:
    """Each value ``obj`` itself declares, an attribute's or a parameter default, by where it is."""
    found: dict[str, tuple[griffe.Object, object]] = {}
    for name, member in obj.members.items():
        if member.is_alias:
            # a re-exported constant, such as a group name agent_env.plugins exports
            member = _target(member) if obj.is_module and member.is_public else None
            if member is None or not member.is_attribute:
                continue
        if member.is_attribute and member.value is not None:
            found[name] = (member, member.value)
        elif member.is_function:
            found.update({f"{name}({p.name})": (member, p.default) for p in member.parameters if p.default is not None})
    return found


def value_drift(path: str, old: griffe.Object, new: griffe.Object) -> list[str]:
    """Values written the same way whose constant changed, such as a default naming a module-level
    TTL that went from 10800 to 60; griffe compares the text, so it sees no change."""
    found = []
    after = _values(new)
    for where, (old_scope, value) in _values(old).items():
        if where not in after or str(after[where][1]) != str(value) or not isinstance(value, griffe.Expr):
            continue
        before, now = _literal(old_scope, value), _literal(after[where][0], after[where][1])
        if before is not _UNREADABLE and now is not _UNREADABLE and before != now:
            found.append(f"{path}.{where}: {value} changed value: {before!r} -> {now!r}")
    return found


def construction(path: str, old: griffe.Class, new: griffe.Class) -> list[str]:
    """How instances are built: the dataclass options and a pydantic ``model_config``."""
    found = []

    def dataclass(cls: griffe.Class) -> list[str]:
        return sorted(str(d.value) for d in cls.decorators if "dataclass" in str(d.value))

    if dataclass(old) != dataclass(new):
        found.append(f"{path}: dataclass options changed: {dataclass(old) or 'none'} -> {dataclass(new) or 'none'}")
    configs = [str(cls.members["model_config"].value) if "model_config" in cls.members else None for cls in (old, new)]
    if configs[0] != configs[1]:
        found.append(f"{path}.model_config: {configs[0] or 'unset'} -> {configs[1] or 'unset'}")
    hook = new.members.get("__init_subclass__")
    if hook is not None and "__init_subclass__" not in old.members and _raises(hook):
        found.append(f"{path}.__init_subclass__: new, and it can raise, which refuses a subclass")
    metaclasses = [str(getattr(cls, "keywords", {}).get("metaclass", "")) for cls in (old, new)]
    if metaclasses[0] != metaclasses[1]:
        found.append(f"{path}: metaclass changed: {metaclasses[0] or 'none'} -> {metaclasses[1] or 'none'}")
    return found


def _source(fn: griffe.Object) -> ast.stmt | None:
    try:
        lines = Path(fn.filepath).read_text().splitlines()[fn.lineno - 1:fn.endlineno]
        return ast.parse(textwrap.dedent("\n".join(lines))).body[0]
    except (OSError, SyntaxError, IndexError, TypeError):
        return None


def _raises(fn: griffe.Object) -> bool:
    """Whether ``fn`` has a ``raise`` of its own; unreadable source counts as one."""
    node = _source(fn)
    return node is None or any(isinstance(n, ast.Raise) for n in ast.walk(node))


def _only_raises_not_implemented(fn: griffe.Function) -> bool:
    node = _source(fn)
    if node is None:
        return False
    body = [n for n in getattr(node, "body", []) if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    if len(body) != 1 or not isinstance(body[0], ast.Raise) or body[0].exc is None:
        return False
    exc = body[0].exc.func if isinstance(body[0].exc, ast.Call) else body[0].exc
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


def new_members(path: str, old: griffe.Class, new: griffe.Class) -> tuple[list[str], list[str]]:
    """New members a subclass may have to provide: a ``ClassVar`` with no value is a break; a stub
    that only raises NotImplementedError and an annotation with no value get a note, since whether
    core requires them cannot be told statically."""
    found, notes = [], []
    for name, member in new.members.items():
        if name in old.all_members or member.is_alias or not member.is_public:
            continue
        if member.is_attribute and member.value is None:
            if "class-attribute" in member.labels:  # griffe's label for a ClassVar annotation
                found.append(f"{path}.{name}: a new ClassVar with no value; every subclass must now set it")
            else:
                notes.append(f"{path}.{name}: a new attribute with no value; if every subclass must set it, "
                             "that is a break")
        elif name == "__init_subclass__" and not _raises(member):
            notes.append(f"{path}.__init_subclass__: new; it runs on every subclass, but raises nothing of its own")
        elif member.is_function and "abstractmethod" not in member.labels and _only_raises_not_implemented(member):
            notes.append(f"{path}.{name}: a new method that only raises NotImplementedError; if every subclass "
                         "must implement it, declare it @abstractmethod or in _MUST_IMPLEMENT so this check "
                         "enforces it")
    return found, notes


def breaks(old: griffe.Module, new: griffe.Module) -> tuple[list[str], list[str]]:
    """The breaks to the surface from ``old`` to ``new``, and notes that do not fail the check."""
    (old_declared, old_unreadable), (new_declared, new_unreadable) = declared_required(old), declared_required(new)
    found: list[str] = list(new_unreadable)
    notes: list[str] = []
    for path in SURFACE:
        before = lookup(old, path)
        if before is None:
            continue
        after = lookup(new, path)
        if after is None or after.kind is not before.kind:
            found.append(f"{path}: " + ("removed" if after is None else f"{before.kind.value} -> {after.kind.value}"))
            continue
        if path in BASES and not after.is_class:
            found.append(f"{path}: not a class, so it cannot be checked; update BASES in {Path(__file__).name}")
            continue
        reshaped: set[str] = set()
        if path in BASES:
            old_required, new_required = required(before, old_declared), required(after, new_declared)
            if old_unreadable:
                # nothing to compare the declared part with, so assume it did not change
                old_required |= new_declared.get(after.path, set())
            for cls in (before, after):
                for name in old_required | new_required:
                    if name in cls.all_members and (target := _target(cls.all_members[name])) is not None:
                        target.public = True
            subclass, reshaped = subclass_breaks(path, before, after, old_required, new_required)
            found += subclass + construction(path, before, after)
            added, noted = new_members(path, before, after)
            found += added
            notes += noted
        elif before.is_class:
            found += construction(path, before, after)
        found += caller_breaks(path, before, after, reshaped)
        found += value_drift(path, before, after)
    return list(dict.fromkeys(found)), list(dict.fromkeys(notes))


def archive(ref: str, into: Path) -> Path:
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", ref, "src/agent_env",
                          ":(exclude)src/agent_env/explorer/ui"], capture_output=True)
    if tar.returncode:
        raise SystemExit(f"plugin API: git archive {ref} failed: {tar.stderr.decode().strip()}; "
                         "the checkout must include the base commit")
    with tarfile.open(fileobj=io.BytesIO(tar.stdout)) as extract:
        extract.extractall(into, filter="data")
    return into / "src"


def _report(level: str, title: str, lines: list[str]) -> None:
    """In GitHub Actions, an annotation on the run and a section in its summary."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    print(f"::{level} title=Plugin API::{title}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as out:
            out.write(f"### Plugin API\n\n{title}\n\n" + "".join(f"- `{line}`\n" for line in lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = parser.add_mutually_exclusive_group(required=True)
    base.add_argument("--base", metavar="REF", help="the git ref to compare with")
    base.add_argument("--old", type=Path, metavar="DIR", help="a directory holding the old agent_env package")
    parser.add_argument("--new", type=Path, default=ROOT / "src", metavar="DIR",
                        help="a directory holding the new agent_env package (default: this checkout's)")
    parser.add_argument("--title", help="the pull request or commit title; a marked break passes")
    parser.add_argument("--on-main", action="store_true",
                        help="the title is a commit already on main, so report rather than advise")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        found, notes = breaks(load(args.old or archive(args.base, Path(tmp))), load(args.new))
    for line in notes:
        print(f"plugin API: note: {line}")
    if notes:
        _report("notice", f"{len(notes)} note{'s' if len(notes) > 1 else ''} on the plugin surface; notes do not fail "
                "the check", notes)
    if not found:
        print("plugin API: no break to the plugin surface")
        return 0
    count = f"{len(found)} break{'s' if len(found) > 1 else ''} to the plugin surface"
    print(f"plugin API: {count}:")
    for line in found:
        print(f"  {line}")
    title = args.title or ""
    if MARKED.match(title):
        print(f"plugin API: the title marks the break: {title}")
        _report("warning", f"{count}, marked in the title", found)
        return 0
    if args.on_main:
        print("plugin API: this reached main without a ! in its title")
        _report("error", f"{count} reached main without a ! in its title", found)
        return 1
    if misplaced := MISPLACED.match(title):
        print(f"plugin API: the ! goes after the scope: {misplaced[1]}{misplaced[2]}!: …")
    elif "!" in title:
        print("plugin API: the title has a ! but not in the form type(scope)!: summary, with the type one of "
              + ", ".join(TYPES))
    print("plugin API: if the break is deliberate, mark it with ! in the pull request title, as in "
          "feat(plugins)!: …; see \"Plugin compatibility\" in README.md")
    _report("error", f"{count}; mark it with ! in the title if it is deliberate", found)
    return 1


if __name__ == "__main__":
    sys.exit(main())
