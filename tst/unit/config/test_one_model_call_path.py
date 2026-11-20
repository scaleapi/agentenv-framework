"""Core calls a model itself only through ``config/model_call.py``: no other module in ``src/agent_env`` imports
LiteLLM. Eval integrity and the record of each judge call's model, host and cost rest on no framework call going
around it. A shipped agent's folder (``agents/agents/``) isn't core: it runs in its own image with its own client."""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "agent_env"
THE_PATH = "config/model_call.py"
NOT_CORE = ("agents/agents/",)


def _imports_litellm(tree: ast.AST) -> bool:
    """Whether ``tree`` imports LiteLLM or a module of it, anywhere: by statement, or by ``import_module`` or
    ``__import__`` with the name written out."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "litellm" for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and not node.level and (node.module or "").split(".")[0] == "litellm":
            return True
        if (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str) and node.args[0].value.split(".")[0] == "litellm"
                and getattr(node.func, "attr", getattr(node.func, "id", None)) in {"import_module", "__import__"}):
            return True
    return False


def test_only_the_one_model_call_imports_litellm():
    offenders = sorted(
        rel for path in SRC.rglob("*.py")
        if (rel := path.relative_to(SRC).as_posix()) != THE_PATH and not rel.startswith(NOT_CORE)
        and _imports_litellm(ast.parse(path.read_text()))
    )

    assert offenders == [], (
        f"{offenders} import LiteLLM. Core calls a model only through {THE_PATH} (route, complete), which records "
        "what each call called and what it cost; call it from there.")
    assert _imports_litellm(ast.parse((SRC / THE_PATH).read_text()))


@pytest.mark.parametrize("source", [
    "import litellm", "import litellm.cost_calculator as c", "from litellm import acompletion",
    "from litellm.types.utils import ModelResponse", "def f():\n    import litellm", "importlib.import_module('litellm')",
    "__import__('litellm.utils')",
])
def test_the_guard_sees_each_way_of_importing_litellm(source):
    assert _imports_litellm(ast.parse(source))


@pytest.mark.parametrize("source", [
    "import litellm_helper", "from .litellm import x", "from . import litellm", "litellm_api_key = 'k'",
    "importlib.import_module(name)",
])
def test_the_guard_leaves_other_names_alone(source):
    assert not _imports_litellm(ast.parse(source))
