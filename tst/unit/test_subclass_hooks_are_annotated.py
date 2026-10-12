"""agent_env ships py.typed, so a consumer's type checker checks its subclasses of agent_env's classes, and defining one calls
the base's ``__init_subclass__``, which a strict checker refuses to call when it's untyped (``no-untyped-call``). Every such
hook is annotated."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "agent_env"


def _unannotated(hook: ast.FunctionDef) -> list[str]:
    args = hook.args
    after_cls = [*args.posonlyargs, *args.args][1:]
    params = [*after_cls, *args.kwonlyargs, *filter(None, (args.vararg, args.kwarg))]
    return [param.arg for param in params if param.annotation is None] + (["return"] if hook.returns is None else [])


def test_every_init_subclass_hook_is_annotated():
    unannotated = {
        f"{path.relative_to(SRC)}:{node.lineno}": missing
        for path in sorted(SRC.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "__init_subclass__" and (missing := _unannotated(node))
    }
    assert unannotated == {}
