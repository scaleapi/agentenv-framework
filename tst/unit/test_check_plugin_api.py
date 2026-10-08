"""The plugin API check in ``.github/scripts/check_plugin_api.py``, on small before and after trees."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / ".github" / "scripts" / "check_plugin_api.py"
_spec = importlib.util.spec_from_file_location("check_plugin_api", SCRIPT)
check_plugin_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_plugin_api)

BASE = {
    "env/env.py": '''
from abc import ABC

DEFAULT_TIMEOUT = 30


class Env(ABC):
    """An environment."""

    type: str = "env"
    description: str

    def __init__(self, id, version=None):
        self.id = id
        self.version = version

    @classmethod
    def from_dict(cls, data):
        raise NotImplementedError

    async def deploy(self, **kwargs):
        return None

    async def reset(self, deployed, *, timeout=DEFAULT_TIMEOUT):
        return None

    def to_dict(self):
        return {"id": self.id}

    def _helper(self):
        return None
''',
    "env/registry.py": '''
from agent_env.env.env import Env
from agent_env.plugins._registration import unimplemented

_MUST_IMPLEMENT = ("from_dict",)


def check(cls):
    return unimplemented(cls, Env, _MUST_IMPLEMENT)
''',
    "providers/env_state/env_state_provider.py": '''
from abc import ABC, abstractmethod


class EnvStateProvider(ABC):
    @abstractmethod
    async def acquire(self, ctx):
        ...

    async def teardown(self, instance):
        await self._teardown(instance)

    @abstractmethod
    async def _teardown(self, instance):
        ...
''',
    "artifact/artifact.py": '''
from pydantic import BaseModel, Field


class Artifact(BaseModel):
    id: str = Field(description="Unique id")
    version: int = Field(default=0, description="The version")
''',
    "task_step/context.py": '''
from dataclasses import dataclass, field


@dataclass
class TaskStepContext:
    metadata: dict = field(default_factory=dict)
    agent_model: str | None = None
''',
    "plugins/__init__.py": '''
from agent_env.plugins._registration import ENVS, load_failures
from agent_env.plugins.tables import settings

__all__ = ["ENVS", "load_failures", "settings"]
''',
    "plugins/_registration.py": '''
ENVS = "agent_env.envs"


def load_failures(config=None):
    return {}


def unimplemented(cls, base, required=()):
    return None
''',
    "plugins/tables.py": '''
def settings(package, config=None):
    return {}
''',
    "internal.py": '''
def compute(x):
    return x
''',
}

ENV, REGISTRY, STATE, CONTEXT, PLUGINS, ARTIFACT = (
    "env/env.py", "env/registry.py", "providers/env_state/env_state_provider.py", "task_step/context.py",
    "plugins/__init__.py", "artifact/artifact.py",
)


def _write(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / "agent_env" / relative
        for parent in [path.parent, *path.parent.parents]:
            if parent == root:
                break
            parent.mkdir(parents=True, exist_ok=True)
            (parent / "__init__.py").touch()
        path.write_text(text)
    return root


def _check(tmp_path: Path, edits: list[tuple[str, str, str]], title: str | None = None) -> subprocess.CompletedProcess:
    """Run the check on ``BASE`` against a copy with each ``(file, old, new)`` replacement made."""
    new = dict(BASE)
    for relative, old, replacement in edits:
        assert old in new[relative], old
        new[relative] = new[relative].replace(old, replacement)
    command = [sys.executable, str(SCRIPT), "--old", str(_write(tmp_path / "old", BASE)),
               "--new", str(_write(tmp_path / "new", new))]
    return subprocess.run(command + (["--title", title] if title is not None else []), capture_output=True, text=True)


@pytest.mark.parametrize(("edits", "reported"), [
    ([(ENV, "    def to_dict(self):\n        return {\"id\": self.id}\n", "")],
     "agent_env.env.env.Env.to_dict: Public object was removed"),
    ([(ENV, "def reset(self, deployed,", "def reset(self, instance,")],
     "agent_env.env.env.Env.reset(deployed): Parameter was removed"),
    ([(ENV, "def to_dict(self):", "def to_dict(self, fields):")],
     "agent_env.env.env.Env.to_dict(fields): Parameter was added as required"),
    ([(ENV, "timeout=DEFAULT_TIMEOUT", "timeout=60")],
     "agent_env.env.env.Env.reset(timeout): Parameter default was changed"),
    ([(ENV, 'type: str = "env"', 'type: str = "environment"')],
     "agent_env.env.env.Env.type: Attribute value was changed"),
    ([(ENV, "class Env(ABC):", "class Env:")], "agent_env.env.env.Env: no longer subclasses abc.ABC"),
    ([(ENV, "from abc import ABC\n", "from abc import ABC, abstractmethod\n"),
      (ENV, "    def _helper(self):",
       "    @abstractmethod\n    async def stop(self):\n        ...\n\n    def _helper(self):")],
     "agent_env.env.env.Env.stop: every subclass must now implement it (new)"),
    ([(ENV, "from abc import ABC\n", "from abc import ABC, abstractmethod\n"),
      (ENV, "    def to_dict(self):", "    @abstractmethod\n    def to_dict(self):")],
     "agent_env.env.env.Env.to_dict: every subclass must now implement it"),
    ([(REGISTRY, '("from_dict",)', '("from_dict", "to_dict")')],
     "agent_env.env.env.Env.to_dict: every subclass must now implement it"),
    ([(ENV, "async def reset(self, deployed, *, timeout=DEFAULT_TIMEOUT):",
       "async def reset(self, deployed, *, timeout=DEFAULT_TIMEOUT, force=False):")],
     "agent_env.env.env.Env.reset(force): new parameter"),
    ([(ENV, "async def reset(", "def reset(")], "agent_env.env.env.Env.reset: async function -> function"),
    ([(ENV, "    @classmethod\n    def from_dict(cls, data):", "    @staticmethod\n    def from_dict(data):")],
     "agent_env.env.env.Env.from_dict: classmethod function -> staticmethod function"),
    ([(STATE, "async def _teardown(self, instance):", "async def _teardown(self, instance, force):")],
     "EnvStateProvider._teardown(force): Parameter was added as required"),
    ([(CONTEXT, "    agent_model: str | None = None\n", "")],
     "agent_env.task_step.context.TaskStepContext.agent_model: Public object was removed"),
    ([(PLUGINS, '"ENVS", ', "")], "agent_env.plugins.ENVS: no longer in __all__"),
    ([(PLUGINS, ', "load_failures"', ""), (PLUGINS, "import ENVS, load_failures", "import ENVS")],
     "agent_env.plugins.load_failures: Public object was removed"),
    ([("plugins/tables.py", "def settings(package, config=None):", "def settings(package, *, config=None):")],
     "agent_env.plugins.settings(config): Parameter kind was changed"),
    ([("plugins/_registration.py", 'ENVS = "agent_env.envs"', 'ENVS = "agent_env.environments"')],
     "agent_env.plugins.ENVS: Attribute value was changed"),
    ([(ARTIFACT, "Field(default=0,", "Field(default=1,")],
     "agent_env.artifact.artifact.Artifact.version: Attribute value was changed"),
], ids=["method-removed", "parameter-renamed", "required-parameter-added", "default-changed", "class-attribute-changed",
        "base-removed", "abstract-method-added", "method-became-abstract", "method-became-required",
        "hook-parameter-added", "async-to-sync", "classmethod-to-staticmethod", "required-private-method-changed",
        "dataclass-field-removed", "dropped-from-all", "export-removed", "settings-signature-changed",
        "group-name-changed", "field-default-changed"])
def test_reports_a_break(tmp_path, edits, reported):
    result = _check(tmp_path, edits)

    assert result.returncode == 1, result.stdout + result.stderr
    assert reported in result.stdout, result.stdout


@pytest.mark.parametrize("edits", [
    [(ENV, '"""An environment."""', '"""An environment a task deploys."""')],
    [(ENV, "    def _helper(self):",
      "    def status(self, verbose=False):\n        return None\n\n    def _helper(self):")],
    [(ENV, "def _helper(self):", "def _assist(self):")],
    [("internal.py", "def compute(x):", "def compute(x, y):")],
    [(ENV, "async def deploy(self, **kwargs):", "async def deploy(self, *, region=None, **kwargs):")],
    [(ENV, "    description: str\n", '    description: str = "An environment"\n')],
    [(ENV, "        self.version = version\n", "        self.version = version or 0\n")],
    [(ENV, "timeout=DEFAULT_TIMEOUT", "timeout=30")],
    [(CONTEXT, "    agent_model: str | None = None\n",
      "    agent_harness: str | None = None\n    agent_model: str | None = None\n")],
    [(ARTIFACT, 'Field(description="Unique id")', 'Field(description="The artifact\'s id")')],
], ids=["docstring", "new-optional-method", "private-rename", "internal-module", "parameter-swallowed-by-kwargs",
        "annotation-gains-a-value", "instance-attribute-expression", "default-to-equal-literal",
        "dataclass-field-inserted", "field-description"])
def test_passes_a_compatible_change(tmp_path, edits):
    result = _check(tmp_path, edits)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no break" in result.stdout


REMOVE_TO_DICT = [(ENV, "    def to_dict(self):\n        return {\"id\": self.id}\n", "")]


@pytest.mark.parametrize("title", ["feat(env)!: drop Env.to_dict", "refactor!: drop Env.to_dict"])
def test_a_title_marked_with_a_bang_passes_and_lists_the_breaks(tmp_path, title):
    result = _check(tmp_path, REMOVE_TO_DICT, title)

    assert result.returncode == 0, result.stdout
    assert "Env.to_dict: Public object was removed" in result.stdout and "the title marks the break" in result.stdout


@pytest.mark.parametrize(("title", "hint"), [
    ("feat(env): drop Env.to_dict", "mark it with !"),
    ("refactor!(env): drop Env.to_dict", "the ! goes after the scope: refactor(env)!: "),
    ("feat(env) !: drop Env.to_dict", "mark it with !"),
])
def test_an_unmarked_break_fails(tmp_path, title, hint):
    result = _check(tmp_path, REMOVE_TO_DICT, title)

    assert result.returncode == 1 and hint in result.stdout, result.stdout


def test_a_removed_surface_class_is_a_break(tmp_path):
    result = _check(tmp_path, [(STATE, "class EnvStateProvider(ABC):", "class StateProvider(ABC):")])

    assert result.returncode == 1
    assert "agent_env.providers.env_state.env_state_provider.EnvStateProvider: removed" in result.stdout


def _documented_types(path: str, lead: str) -> set[str]:
    text = " ".join((REPO / path).read_text().split())
    listed = text.split(lead, 1)[1].split(".", 1)[0].split(";", 1)[0]
    return set(re.findall(r"`([a-z]+)`", listed))


@pytest.mark.parametrize(("path", "lead"), [("CONTRIBUTING.md", "Types in use:"), ("AGENTS.md", "with the types in use")])
def test_the_title_types_are_the_documented_ones(path, lead):
    assert _documented_types(path, lead) == set(check_plugin_api.TYPES)


def test_the_surface_is_found_in_this_checkout():
    pkg = check_plugin_api.load(REPO / "src")
    found = {path: check_plugin_api.lookup(pkg, path) for path in check_plugin_api.SURFACE}

    assert all(found[path] is not None and found[path].is_class for path in check_plugin_api.BASES), found
    assert found["agent_env.plugins"].is_module
    declared, unreadable = check_plugin_api.declared_required(pkg)
    assert unreadable == []
    assert "from_dict" in declared["agent_env.env.env.Env"]
    assert "from_dict" in declared["agent_env.task_step.task_step.TaskStep"]
    assert check_plugin_api.breaks(pkg, check_plugin_api.load(REPO / "src")) == ([], [])


@pytest.mark.parametrize(("edits", "reported"), [
    ([(ENV, 'type: str = "env"', "type: str")], "agent_env.env.env.Env.type: Attribute value was changed"),
    ([(ENV, "def reset(self, deployed,", "def reset(self, deployed=None,")],
     "agent_env.env.env.Env.reset(deployed): no longer required"),
    ([(ENV, "def to_dict(self):", "def to_dict(self, **options):")], "agent_env.env.env.Env.to_dict: now takes **options"),
    ([(ENV, "def reset(self, deployed, *, timeout=DEFAULT_TIMEOUT):", "def reset(self, deployed, timeout=DEFAULT_TIMEOUT):")],
     "agent_env.env.env.Env.reset(timeout): can now be passed by position"),
    ([(REGISTRY, "unimplemented(cls, Env, _MUST_IMPLEMENT)", 'unimplemented(cls, base=Env, required=("from_dict", "to_dict"))')],
     "agent_env.env.env.Env.to_dict: every subclass must now implement it"),
    ([(ENV, "DEFAULT_TIMEOUT = 30", "DEFAULT_TIMEOUT = 60")],
     "agent_env.env.env.Env.reset(timeout): DEFAULT_TIMEOUT changed value: 30 -> 60"),
    ([(CONTEXT, "@dataclass\nclass", "@dataclass(frozen=True)\nclass")],
     "agent_env.task_step.context.TaskStepContext: dataclass options changed"),
    ([(ARTIFACT, "from pydantic import BaseModel, Field", "from pydantic import BaseModel, ConfigDict, Field"),
      (ARTIFACT, "class Artifact(BaseModel):\n", "class Artifact(BaseModel):\n    model_config = ConfigDict(frozen=True)\n")],
     "agent_env.artifact.artifact.Artifact.model_config: unset -> ConfigDict(frozen=True)"),
    ([(ENV, "from abc import ABC\n", "from abc import ABC\nfrom typing import ClassVar\n"),
      (ENV, "    description: str\n", "    description: str\n    category: ClassVar[str]\n")],
     "agent_env.env.env.Env.category: a new ClassVar with no value; every subclass must now set it"),
    ([(PLUGINS, '__all__ = ["ENVS", "load_failures", "settings"]', "__all__ = build_all()")],
     "agent_env.plugins.__all__: cannot read it"),
    ([(ENV, "    def _helper(self):", "    def __init_subclass__(cls, **kwargs):\n        super().__init_subclass__(**kwargs)\n"
       "        if not cls.__doc__:\n            raise TypeError(cls)\n\n    def _helper(self):")],
     "agent_env.env.env.Env.__init_subclass__: new, and it can raise"),
    ([(ENV, "class Env(ABC):", "class _Strict(type):\n    pass\n\n\nclass Env(ABC, metaclass=_Strict):")],
     "agent_env.env.env.Env: metaclass changed"),
    ([("plugins/_registration.py", 'ENVS = "agent_env.envs"', '_PREFIX = "agent_env."\nENVS = _PREFIX + "environments"')],
     "agent_env.plugins.ENVS"),
    ([(REGISTRY, '_MUST_IMPLEMENT = ("from_dict",)', "_MUST_IMPLEMENT = compute_required()")],
     "cannot read the unimplemented() call"),
], ids=["attribute-lost-its-value", "hook-parameter-became-optional", "hook-gained-kwargs", "keyword-only-became-positional",
        "required-given-by-keyword", "constant-behind-a-default-changed", "dataclass-became-frozen", "model-config-added",
        "classvar-without-a-value", "all-unreadable", "init-subclass-added", "metaclass-added",
        "group-name-through-an-expression", "declaration-unreadable"])
def test_reports_a_subclass_or_construction_break(tmp_path, edits, reported):
    result = _check(tmp_path, edits)

    assert result.returncode == 1, result.stdout + result.stderr
    assert reported in result.stdout, result.stdout


@pytest.mark.parametrize("edits", [
    [(ENV, "from abc import ABC\n", "import abc\n"), (ENV, "class Env(ABC):", "class Env(abc.ABC):")],
    [(ENV, 'type: str = "env"', 'type: str = "e" + "nv"')],
    [(REGISTRY, '_MUST_IMPLEMENT = ("from_dict",)', '_MUST_IMPLEMENT = frozenset({"from_dict"})')],
    [(REGISTRY, '_MUST_IMPLEMENT = ("from_dict",)', '_BASE = ("from_dict",)\n_MUST_IMPLEMENT = (*_BASE,)')],
    [(PLUGINS, '__all__ = ["ENVS", "load_failures", "settings"]', '__all__ = sorted(["ENVS", "load_failures", "settings"])')],
    [("plugins/_registration.py", 'ENVS = "agent_env.envs"', '_PREFIX = "agent_env."\nENVS = _PREFIX + "envs"')],
], ids=["base-respelled", "equal-value-rewritten", "declaration-as-frozenset", "declaration-starred", "all-sorted",
        "group-name-as-an-equal-expression"])
def test_passes_an_equivalent_rewrite(tmp_path, edits):
    result = _check(tmp_path, edits)

    assert result.returncode == 0, result.stdout + result.stderr


def test_an_unreadable_declaration_passes_when_marked(tmp_path):
    result = _check(tmp_path, [(REGISTRY, '_MUST_IMPLEMENT = ("from_dict",)', "_MUST_IMPLEMENT = compute_required()")],
                    "refactor(env)!: compute what subclasses implement")

    assert result.returncode == 0, result.stdout + result.stderr


def test_a_new_init_subclass_that_raises_nothing_gets_a_note_and_passes(tmp_path):
    result = _check(tmp_path, [(ENV, "    def _helper(self):",
                                "    def __init_subclass__(cls, **kwargs):\n        super().__init_subclass__(**kwargs)\n"
                                "        cls.refs = None\n\n    def _helper(self):")])

    assert result.returncode == 0, result.stdout
    assert "note: agent_env.env.env.Env.__init_subclass__: new; it runs on every subclass" in result.stdout


def test_a_new_not_implemented_stub_gets_a_note_and_passes(tmp_path):
    result = _check(tmp_path, [(ENV, "    def _helper(self):",
                                "    def snapshot(self):\n        raise NotImplementedError\n\n    def _helper(self):")])

    assert result.returncode == 0, result.stdout
    assert "note: agent_env.env.env.Env.snapshot: a new method that only raises NotImplementedError" in result.stdout


@pytest.mark.parametrize(("title", "code", "hint"), [
    ("FEAT(env)!: drop Env.to_dict", 0, "the title marks the break"),
    ("wip!: drop Env.to_dict", 1, "has a ! but not in the form type(scope)!: summary"),
    ("feat()!: drop Env.to_dict", 1, "has a ! but not in the form"),
])
def test_title_forms(tmp_path, title, code, hint):
    result = _check(tmp_path, REMOVE_TO_DICT, title)

    assert result.returncode == code and hint in result.stdout, result.stdout


def test_on_main_reports_rather_than_advises(tmp_path):
    new = dict(BASE)
    new[ENV] = new[ENV].replace("    def to_dict(self):\n        return {\"id\": self.id}\n", "")
    command = [sys.executable, str(SCRIPT), "--old", str(_write(tmp_path / "old", BASE)),
               "--new", str(_write(tmp_path / "new", new)), "--title", "feat(env): drop Env.to_dict", "--on-main"]
    result = subprocess.run(command, capture_output=True, text=True)

    assert result.returncode == 1 and "reached main without a ! in its title" in result.stdout, result.stdout
