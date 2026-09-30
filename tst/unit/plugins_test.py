"""Entry-point plugins for the six type registries (``agent_env.plugins``).

Covers the contract: the entry-point name is the registry key; a plugin cannot replace a
built-in; a name two distributions claim is left out, naming both, while the rest of its group
loads, and config cannot take it; a broken plugin, or one whose class leaves a method its base
requires unimplemented, is skipped and its name then says why; config
replaces a plugin's class with a warning (silently when it names the same class, which is the
transitional sdk shape); and the provider registries' three-case split, so a config table for a
plugin's name patches it instead of colliding.
"""

import asyncio
import importlib
from abc import abstractmethod
import logging
import sys
import textwrap
from importlib.metadata import EntryPoint
from typing import Literal

import dataclasses

import pytest
from fastapi import APIRouter

from agent_env import plugins
from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.store import ArtifactStore
from agent_env.config import ConfigError, get_config, load_impl, reset_config
from agent_env.env.env import DeployedEnv, DeployedSandboxEnv, Env
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.registry import _MUST_IMPLEMENT as _ENV_MUST_IMPLEMENT
from agent_env.explorer.app import create_app
from agent_env.explorer.plugin import ExplorerPlugin, load_plugins
from agent_env.plugins import _discovery, _registration
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.providers.env_providers.env_provider import EnvironmentProvider, build_env_provider
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SandboxProvider,
    SandboxProviderTypeError,
    build_sandbox_provider,
    get_sandbox_provider,
)
from agent_env.providers.env_state.env_state_provider import EnvStateProvider, build_state_provider
from agent_env.task.task import Task
from agent_env.task_step.registry import _MUST_IMPLEMENT as _STEP_MUST_IMPLEMENT
from agent_env.task_step.task_step import TaskStep
from tst.unit.providers.env_state.test_config_state_providers import _RecordingStateProvider
from tst.unit.providers.sandbox_providers.test_config_sandbox_providers import (
    _MintingProvider,
    _RecordingProvider,
)

_HERE = "tst.unit.plugins_test"


class _Env(Env):
    """What an env plugin has to implement, and nothing more."""

    @classmethod
    def from_dict(cls, data: dict) -> "_Env":
        return cls(data["id"], data.get("version"))


class _Step(TaskStep):
    """What a task step plugin has to implement, and nothing more."""

    async def execute(self, context):
        return context

    @classmethod
    def from_dict(cls, data: dict) -> "_Step":
        return cls(**cls._base_from_dict(data))


class _BrowserEnv(_Env):
    type = "browser"


class _OtherBrowserEnv(_Env):
    type = "browser"


class _UntypedEnv(_Env):
    pass


class _PluginStep(_Step):
    type = "plugin_step"


class _PluginArtifact(Artifact):
    type: Literal["plugin_art", "plugin_art_legacy"] = "plugin_art"


class _OtherPluginArtifact(Artifact):
    type: Literal["plugin_art", "plugin_art_legacy"] = "plugin_art"


class _StrictArtifact(Artifact):
    type: Literal["strict_art"] = "strict_art"


class _SiblingEnv(_Env):
    type = "sibling"


class _SiblingStep(_Step):
    type = "sibling"


class _SiblingArtifact(Artifact):
    type: Literal["sibling"] = "sibling"


class _MyWebsite(WebsiteEnv):
    pass


class _MyFile(FileArtifact):
    pass


class _PluginState(_RecordingStateProvider):
    type = "plugin_state"


class _OtherPluginState(_RecordingStateProvider):
    type = "plugin_state"


class _OtherRecordingProvider(_RecordingProvider):
    pass


class _GuardedProvider(_MintingProvider):
    pass


class _Routes(ExplorerPlugin):
    type = "plugin_routes"

    @property
    def router(self) -> APIRouter:
        return APIRouter()


class _OtherRoutes(_Routes):
    pass


class _SiblingRoutes(_Routes):
    type = "sibling"


class _SiblingState(_RecordingStateProvider):
    type = "sibling"


class _PluginEnvProvider(EnvironmentProvider):
    type = "plugin_env"

    async def deploy(self, env, sandbox_provider):
        return DeployedEnv(env_id=env.id, env_version=env.version, env_provider_type=self.type)

    async def close(self):
        pass


class _OtherPluginEnvProvider(_PluginEnvProvider):
    pass


class _SiblingEnvProvider(_PluginEnvProvider):
    type = "sibling"


class _CountingRoutes(ExplorerPlugin):
    """A router property that builds a fresh router per read, which the contract allows."""

    type = "counting_routes"
    reads = 0

    @property
    def router(self) -> APIRouter:
        type(self).reads += 1
        return APIRouter()


class _BrokenCtorRoutes(ExplorerPlugin):
    type = "broken_routes"

    def __init__(self):
        raise RuntimeError("boom in ctor")

    @property
    def router(self) -> APIRouter:
        return APIRouter()


# Each leaves unimplemented what its group requires, as a first attempt at a plugin does.
class _BareEnv(Env):
    type = "bare"


class _BareStep(TaskStep):
    type = "bare"


class _BareArtifact(Artifact):
    type: Literal["bare"] = "bare"

    @abstractmethod
    def render(self) -> str: ...


class _BareProvider(SandboxProvider):
    pass


class _BareState(EnvStateProvider):
    type = "bare"


class _BareRoutes(ExplorerPlugin):
    type = "bare"


class _BareEnvProvider(EnvironmentProvider):
    type = "bare"


class _UnboundStep(_Step):
    type = "unbound"

    def from_dict(cls, data):  # the @classmethod left off
        return cls(**cls._base_from_dict(data))


class _StaticStep(_Step):
    type = "static"

    @staticmethod
    def from_dict(data):
        return _StaticStep(**TaskStep._base_from_dict(data))


class _Dist:
    def __init__(self, name: str, version: str, requires: list[str] | None = None):
        self.name = name
        self.version = version
        self.requires = requires


class _EP:
    """An installed entry point: name, value and distribution, loaded like the real thing.
    ``on_load`` stands in for the import side effects of the plugin's package."""

    def __init__(self, name: str, attr: str, dist: str = "agentenv-demo", version: str = "1.0", on_load=None,
                 requires: list[str] | None = None):
        self.name = name
        self.value = f"{_HERE}:{attr}"
        self.dist = _Dist(dist, version, requires)
        self._on_load = on_load

    def load(self):
        if self._on_load is not None:
            self._on_load()
        return EntryPoint(self.name, self.value, "unused").load()


class _ReentrantEP(_EP):
    """A plugin whose import asks for its own registry, optionally after ``reset_config()``, as
    a stage bootstrap might; the nested load sees the module half-initialised."""

    def __init__(self, *args, reset: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset = reset
        self.calls = 0

    def load(self):
        self.calls += 1
        if self.calls == 1:
            if self._reset:
                reset_config()
            get_config().env_registry()
        elif self.calls == 2:
            raise AttributeError("partially initialized module 'browser_pkg'")
        return super().load()


def _install(monkeypatch, **groups: list[_EP]) -> None:
    by_group = {f"agent_env.{group}": eps for group, eps in groups.items()}
    monkeypatch.setattr(_discovery, "entry_points", lambda *, group: by_group.get(group, []))


def _use_config(monkeypatch, tmp_path, body: str = "") -> None:
    cfg = tmp_path / ".agentenv" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(textwrap.dedent(body))
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    reset_config()


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    def _reset():
        reset_config()
        sandbox_provider.reset_sandbox_provider()

    monkeypatch.chdir(tmp_path)
    _use_config(monkeypatch, tmp_path)
    _reset()
    yield
    _reset()


def _code(group: str, name: str) -> str | None:
    """The code the inventory reports for the one contribution named ``name`` in ``group``."""
    (found,) = [c for d in plugins.inventory().distributions for c in d.contributions if (c.group, c.name) == (group, name)]
    return found.code


@pytest.fixture
def warnings_logged(caplog):
    caplog.set_level(logging.WARNING)
    return lambda: [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------- registration


def test_the_public_module_is_only_the_contract():
    """Third parties build against agent_env.plugins; anything else it exposed would become contract."""
    assert sorted(plugins.__all__) == [
        "ARTIFACTS", "Claimant", "Contribution", "Diagnostic", "Distribution", "ENVS", "ENV_PROVIDERS", "EXPLORER_PLUGINS",
        "Inventory", "Replacement", "SANDBOX_PROVIDERS", "STATE_PROVIDERS", "Status",
        "TASK_STEPS", "inventory", "load_failures", "settings",
    ]
    assert sorted(n for n in vars(plugins) if not n.startswith("_")) == sorted(plugins.__all__)


def test_every_group_registers_a_plugin_under_its_entry_point_name(monkeypatch):
    _install(
        monkeypatch,
        envs=[_EP("browser", "_BrowserEnv")],
        task_steps=[_EP("plugin_step", "_PluginStep")],
        artifacts=[_EP("plugin_art", "_PluginArtifact")],
        sandbox_providers=[_EP("plugin_box", "_RecordingProvider")],
        state_providers=[_EP("plugin_state", "_PluginState")],
        env_providers=[_EP("plugin_env", "_PluginEnvProvider")],
        explorer_plugins=[_EP("plugin_routes", "_Routes")],
    )
    config = get_config()

    assert config.env_registry()["browser"] is _BrowserEnv
    assert config.task_step_registry()["plugin_step"] is _PluginStep
    assert config.artifact_registry()["plugin_art"] is _PluginArtifact
    assert isinstance(build_sandbox_provider("plugin_box"), _RecordingProvider)
    assert isinstance(build_state_provider("plugin_state"), _PluginState)
    assert isinstance(build_env_provider("plugin_env"), _PluginEnvProvider)
    assert [type(p) for p in load_plugins()] == [_Routes]


def test_one_artifact_class_may_register_under_several_names(monkeypatch):
    _install(monkeypatch, artifacts=[
        _EP("plugin_art", "_PluginArtifact"),
        _EP("plugin_art_legacy", "_PluginArtifact"),
    ])

    registry = get_config().artifact_registry()

    assert registry["plugin_art"] is registry["plugin_art_legacy"] is _PluginArtifact


def test_a_plugin_cannot_replace_a_built_in(monkeypatch, warnings_logged):
    builtin = get_config().env_registry()["mcp_server"]
    reset_config()
    _install(monkeypatch, envs=[_EP("mcp_server", "_BrowserEnv")])

    assert get_config().env_registry()["mcp_server"] is builtin
    assert any("clashes with a built-in" in m for m in warnings_logged())


def test_two_distributions_claiming_a_built_in_name_both_lose_quietly(monkeypatch, warnings_logged):
    builtin = get_config().env_registry()["mcp_server"]
    reset_config()
    _install(monkeypatch, envs=[
        _EP("mcp_server", "_BrowserEnv", dist="agentenv-browser"),
        _EP("mcp_server", "_OtherBrowserEnv", dist="agentenv-web"),
    ])

    assert get_config().env_registry()["mcp_server"] is builtin
    (clash,) = [m for m in warnings_logged() if "clashes with a built-in" in m]
    assert "agentenv-browser" in clash and "agentenv-web" in clash


@pytest.mark.parametrize(("group", "name", "claimants", "sibling", "build", "builtin"), [
    ("envs", "browser", ("_BrowserEnv", "_OtherBrowserEnv"), "_SiblingEnv", lambda c: c.env_registry(), "mcp_server"),
    ("task_steps", "plugin_step", ("_PluginStep", "_PluginStep"), "_SiblingStep", lambda c: c.task_step_registry(),
     "prompt_agent"),
    ("artifacts", "plugin_art", ("_PluginArtifact", "_OtherPluginArtifact"), "_SiblingArtifact",
     lambda c: c.artifact_registry(), "file"),
    ("sandbox_providers", "plugin_box", ("_RecordingProvider", "_OtherRecordingProvider"), "_RecordingProvider",
     lambda c: c.sandbox_registry(), "local"),
    ("state_providers", "plugin_state", ("_PluginState", "_OtherPluginState"), "_SiblingState",
     lambda c: c.state_registry(), "local_postgres"),
    ("env_providers", "plugin_env", ("_PluginEnvProvider", "_OtherPluginEnvProvider"), "_SiblingEnvProvider",
     lambda c: c.env_provider_registry(), "gateway"),
    ("explorer_plugins", "plugin_routes", ("_Routes", "_OtherRoutes"), "_SiblingRoutes",
     lambda c: {p.type: p for p in load_plugins(source=c)}, None),
])
def test_a_name_two_distributions_claim_is_left_out_and_the_rest_of_its_group_loads(
    monkeypatch, warnings_logged, group, name, claimants, sibling, build, builtin
):
    _install(monkeypatch, **{group: [
        _EP(name, claimants[0], dist="agentenv-browser", version="0.1.0"),
        _EP(name, claimants[1], dist="agentenv-web", version="2.1.0"),
        _EP("sibling", sibling, dist="agentenv-browser", version="0.1.0"),
    ]})

    registry = build(get_config())

    assert name not in registry and "sibling" in registry
    assert builtin is None or builtin in registry
    note = _registration.failure_note(f"agent_env.{group}", name)
    assert "agentenv-browser 0.1.0" in note and "agentenv-web 2.1.0" in note and "agent-env plugin remove" in note
    assert plugins.load_failures() == {f"agent_env.{group}": {name: note[len(" ("):-len(")")]}}
    assert sum(f"Plugin name {name!r} in agent_env.{group} was skipped" in m for m in warnings_logged()) == 1


def test_resolving_a_conflicted_name_fails_naming_both_claimants(monkeypatch):
    _install(
        monkeypatch,
        task_steps=[_EP("plugin_step", "_PluginStep", dist="agentenv-a"), _EP("plugin_step", "_PluginStep", dist="agentenv-b")],
        sandbox_providers=[_EP("plugin_box", "_RecordingProvider", dist="agentenv-a"),
                           _EP("plugin_box", "_OtherRecordingProvider", dist="agentenv-b")],
        state_providers=[_EP("plugin_state", "_PluginState", dist="agentenv-a"),
                         _EP("plugin_state", "_OtherPluginState", dist="agentenv-b")],
    )

    for resolve, message in (
        (lambda: Task.from_dict({"id": "t", "steps": [{"id": "s", "type": "plugin_step"}]}), "Unknown task step type"),
        (lambda: build_sandbox_provider("local,plugin_box"), "Unknown sandbox backend"),
        (lambda: build_state_provider("plugin_state"), "Unknown env state type"),
    ):
        with pytest.raises(ValueError, match=message) as raised:
            resolve()
        assert "agentenv-a 1.0" in str(raised.value) and "agentenv-b 1.0" in str(raised.value)
        assert "Remove all but one with `agent-env plugin remove <package>`" in str(raised.value)


def test_a_name_one_package_declares_twice_is_left_out_too(monkeypatch):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv"), _EP("browser", "_OtherBrowserEnv"),
                                _EP("sibling", "_SiblingEnv")])

    registry = get_config().env_registry()

    assert "browser" not in registry and "sibling" in registry
    note = _registration.failure_note(plugins.ENVS, "browser")
    assert "registers 'browser' in agent_env.envs 2 times" in note and "Report it to the package's author" in note
    assert "plugin remove" not in note


def test_one_distribution_found_twice_on_the_path_is_not_a_conflict(monkeypatch):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv"), _EP("browser", "_BrowserEnv")])

    assert get_config().env_registry()["browser"] is _BrowserEnv


# ---------------------------------------------------------------- failure isolation


@pytest.mark.parametrize(("group", "bare", "missing", "sibling", "build"), [
    ("envs", "_BareEnv", "from_dict", "_SiblingEnv", lambda c: c.env_registry()),
    ("task_steps", "_BareStep", "execute and from_dict", "_SiblingStep", lambda c: c.task_step_registry()),
    ("artifacts", "_BareArtifact", "render", "_SiblingArtifact", lambda c: c.artifact_registry()),
    ("sandbox_providers", "_BareProvider", "create_sandbox", "_RecordingProvider", lambda c: c.sandbox_registry()),
    ("state_providers", "_BareState", "_teardown and acquire", "_SiblingState", lambda c: c.state_registry()),
    ("env_providers", "_BareEnvProvider", "close and deploy", "_SiblingEnvProvider", lambda c: c.env_provider_registry()),
    ("explorer_plugins", "_BareRoutes", "router", "_SiblingRoutes",
     lambda c: {p.type: p for p in load_plugins(source=c)}),
])
def test_a_plugin_class_that_leaves_a_required_method_unimplemented_is_skipped(
    monkeypatch, warnings_logged, group, bare, missing, sibling, build
):
    _install(monkeypatch, **{group: [_EP("bare", bare), _EP("sibling", sibling)]})

    registry = build(get_config())

    assert "bare" not in registry and "sibling" in registry
    assert f"{bare} must implement {missing}" in _registration.failure_note(f"agent_env.{group}", "bare")
    assert sum(f"{bare} must implement {missing}" in m for m in warnings_logged()) == 1


def test_resolving_a_type_whose_class_is_unimplemented_says_what_it_lacks(monkeypatch):
    _install(monkeypatch, task_steps=[_EP("bare", "_BareStep")])

    with pytest.raises(ValueError, match=r"Unknown task step type: bare \(.*_BareStep must implement execute and from_dict"):
        Task.from_dict({"id": "t", "steps": [{"id": "s", "type": "bare"}]})


def test_an_implementation_inherited_from_an_intermediate_class_counts():
    assert _registration.unimplemented(_PluginStep, TaskStep, _STEP_MUST_IMPLEMENT) is None
    assert _registration.unimplemented(_MyWebsite, Env, _ENV_MUST_IMPLEMENT) is None
    assert _registration.unimplemented(_BareStep, TaskStep, _STEP_MUST_IMPLEMENT) == (
        "_BareStep must implement execute and from_dict"
    )


def test_a_from_dict_the_class_cannot_call_is_named_and_a_staticmethod_counts(monkeypatch):
    _install(monkeypatch, task_steps=[_EP("unbound", "_UnboundStep"), _EP("static", "_StaticStep")])

    registry = get_config().task_step_registry()

    assert "unbound" not in registry and registry["static"] is _StaticStep
    assert "_UnboundStep.from_dict must be a classmethod" in _registration.failure_note(plugins.TASK_STEPS, "unbound")
    assert Task.from_dict({"id": "t", "steps": [{"id": "s", "type": "static"}]}).steps[0].type == "static"


def test_every_built_in_type_implements_what_a_plugin_of_its_group_must(monkeypatch):
    _install(monkeypatch)
    config = get_config()
    classes = [
        *((cls, Env, _ENV_MUST_IMPLEMENT) for cls in config.env_registry().values()),
        *((cls, TaskStep, _STEP_MUST_IMPLEMENT) for cls in config.task_step_registry().values()),
        *((cls, Artifact, ()) for cls in config.artifact_registry().values()),
        *((load_impl(entry["impl"], SandboxProvider), SandboxProvider, ()) for entry in config.sandbox_registry().values()),
        *((load_impl(entry["impl"], EnvStateProvider), EnvStateProvider, ()) for entry in config.state_registry().values()),
        *((cls, EnvironmentProvider, ()) for cls in config.env_provider_registry().values()),
    ]

    assert [problem for cls, base, required in classes if (problem := _registration.unimplemented(cls, base, required))] == []


@pytest.mark.parametrize(("body", "build", "match"), [
    (f'[envs]\nimpls = ["{_HERE}:_BareEnv"]\n', lambda c: c.env_registry(),
     r"\[envs\] impl '.*:_BareEnv': _BareEnv must implement from_dict"),
    (f'[task_steps]\nimpls = ["{_HERE}:_BareStep"]\n', lambda c: c.task_step_registry(),
     r"\[task_steps\] impl '.*:_BareStep': _BareStep must implement execute and from_dict"),
    (f'[artifacts]\nimpls = ["{_HERE}:_BareArtifact"]\n', lambda c: c.artifact_registry(),
     r"\[artifacts\] impl '.*:_BareArtifact': _BareArtifact must implement render"),
    (f'[sandbox.providers.bare]\nimpl = "{_HERE}:_BareProvider"\n', lambda c: c.sandbox_registry(),
     r"\[sandbox.providers.bare\] impl '.*:_BareProvider': _BareProvider must implement create_sandbox"),
    (f'[state.providers.bare]\nimpl = "{_HERE}:_BareState"\n', lambda c: c.state_registry(),
     r"\[state.providers.bare\] impl '.*:_BareState': _BareState must implement _teardown and acquire"),
    (f'[explorer.plugins]\nimpls = ["{_HERE}:_BareRoutes"]\n', lambda c: load_plugins(source=c),
     r"\[explorer.plugins\] impl '.*:_BareRoutes': _BareRoutes must implement router"),
])
def test_a_config_impl_that_leaves_a_required_method_unimplemented_is_a_config_error(
    monkeypatch, tmp_path, body, build, match
):
    _use_config(monkeypatch, tmp_path, body)

    with pytest.raises(ConfigError, match=match):
        build(get_config())


def test_a_plugin_whose_agent_env_requirement_is_not_met_is_never_imported(monkeypatch, warnings_logged):
    def imported():
        raise AssertionError("the plugin was imported")

    _install(monkeypatch, envs=[
        _EP("browser", "_BrowserEnv", dist="agentenv-future", version="2.0", on_load=imported,
            requires=["agentenv-framework>=999"]),
    ], task_steps=[_EP("plugin_step", "_PluginStep", requires=["agentenv-framework>=0.0.1"])])

    assert "browser" not in get_config().env_registry()
    assert "plugin_step" in get_config().task_step_registry()
    note = _registration.failure_note(plugins.ENVS, "browser")
    assert "agentenv-future 2.0" in note and "but needs agentenv-framework>=999 (installed: " in note
    assert plugins.load_failures() == {plugins.ENVS: {"browser": note[len(" ("):-len(")")]}}
    assert any("was skipped: it needs agentenv-framework>=999" in m for m in warnings_logged())
    assert _code(plugins.ENVS, "browser") == "incompatible-core"


def test_a_plugin_that_cannot_load_does_not_conflict_with_one_that_can(monkeypatch):
    def imported():
        raise AssertionError("the plugin was imported")

    _install(monkeypatch, envs=[
        _EP("browser", "_BrowserEnv", dist="agentenv-browser"),
        _EP("browser", "_OtherBrowserEnv", dist="agentenv-future", on_load=imported, requires=["agentenv-framework>=999"]),
    ])

    assert get_config().env_registry()["browser"] is _BrowserEnv
    assert plugins.load_failures() == {}


def test_a_name_only_plugins_that_cannot_load_claim_says_why_it_is_missing(monkeypatch):
    _install(monkeypatch, envs=[
        _EP("browser", "_BrowserEnv", dist="agentenv-a", requires=["agentenv-framework>=999"]),
        _EP("browser", "_OtherBrowserEnv", dist="agentenv-b", requires=["agentenv-framework-protocol>=999"]),
    ])

    assert "browser" not in get_config().env_registry()
    assert "agentenv-a 1.0" in _registration.failure_note(plugins.ENVS, "browser")


def test_a_broken_plugin_is_skipped_and_its_name_says_why(monkeypatch, warnings_logged):
    broken = _EP("broken_box", "_RecordingProvider", dist="agentenv-broken", version="3.0")
    broken.value = "agentenv_broken.nowhere:Provider"
    _install(monkeypatch, sandbox_providers=[broken, _EP("plugin_box", "_RecordingProvider")])

    assert isinstance(build_sandbox_provider("plugin_box"), _RecordingProvider)
    with pytest.raises(ValueError, match="Unknown sandbox backend") as raised:
        build_sandbox_provider("broken_box")
    assert "agentenv-broken 3.0" in str(raised.value) and "ModuleNotFoundError" in str(raised.value)
    assert any("failed to load" in m for m in warnings_logged())
    assert _code(plugins.SANDBOX_PROVIDERS, "broken_box") == "load-failed"


@pytest.mark.parametrize(
    ("group", "name", "attr", "registry"),
    [
        ("envs", "untyped", "_UntypedEnv", "env_registry"),
        ("envs", "not_an_env", "_PluginStep", "env_registry"),
        ("task_steps", "not_a_step", "_BrowserEnv", "task_step_registry"),
        ("envs", "web_browser", "_BrowserEnv", "env_registry"),
        ("task_steps", "other_step", "_PluginStep", "task_step_registry"),
        ("state_providers", "renamed_state", "_PluginState", "state_registry"),
        ("env_providers", "renamed_env", "_PluginEnvProvider", "env_provider_registry"),
        ("env_providers", "not_a_provider", "_PluginState", "env_provider_registry"),
    ],
)
def test_a_plugin_failing_validation_is_skipped(monkeypatch, group, name, attr, registry):
    _install(monkeypatch, **{group: [_EP(name, attr)]})

    assert name not in getattr(get_config(), registry)()
    assert "failed to load" in _registration.failure_note(f"agent_env.{group}", name)
    assert _code(f"agent_env.{group}", name) == "invalid-plugin"


def test_an_env_is_only_registered_under_the_type_it_writes(monkeypatch):
    """Registered under another name, every env it saved would come back as an unknown type."""
    _install(monkeypatch, envs=[_EP("web_browser", "_BrowserEnv")])

    assert "web_browser" not in get_config().env_registry()
    assert "must be named 'browser'" in _registration.failure_note(plugins.ENVS, "web_browser")


def test_a_plugin_cannot_replace_a_built_in_env_provider(monkeypatch):
    _install(monkeypatch, env_providers=[_EP("gateway", "_PluginEnvProvider")])

    assert get_config().env_provider_registry()["gateway"] is EnvironmentGatewayProvider
    assert _code(plugins.ENV_PROVIDERS, "gateway") == "builtin-name"


def test_building_a_provider_whose_plugin_was_skipped_says_why(monkeypatch):
    _install(monkeypatch, env_providers=[_EP("bare", "_BareEnvProvider")])

    with pytest.raises(ValueError, match=r"Unknown env_provider_type: 'bare' \(expected one of \['gateway', 'server'\]\) "
                                         r"\(.*_BareEnvProvider must implement close and deploy"):
        build_env_provider("bare")


@pytest.mark.parametrize("installed", [True, False], ids=["plugin-installed", "plugin-absent"])
def test_a_plugin_providers_records_load_by_shape_whether_or_not_it_is_installed(monkeypatch, installed):
    _install(monkeypatch, **({"env_providers": [_EP("plugin_env", "_PluginEnvProvider")]} if installed else {}))
    hosted = DeployedEnv(env_id="e", env_version=1, env_provider_type="plugin_env",
                         environment_card_url="https://e.example/.well-known/agent-env.json", environment_card={"name": "e"})
    contained = DeployedSandboxEnv(env_id="e", env_version=1, env_provider_type="plugin_env", sandbox_id="sb-1", sandbox_ids={"e": "sb-1"})

    for record in (hosted, contained):
        loaded = DeployedEnv.from_dict(dataclasses.asdict(record))
        assert (type(loaded), loaded) == (type(record), record)


def test_an_explorer_plugin_named_apart_from_its_type_is_skipped(monkeypatch):
    _install(monkeypatch, explorer_plugins=[_EP("other_routes", "_Routes")])

    assert load_plugins() == []
    assert "must be named 'plugin_routes'" in _registration.failure_note(plugins.EXPLORER_PLUGINS, "other_routes")


def test_an_artifact_legacy_name_needs_the_spelling_the_class_writes(monkeypatch):
    _install(monkeypatch, artifacts=[_EP("plugin_art_legacy", "_PluginArtifact")])

    assert "plugin_art_legacy" not in get_config().artifact_registry()
    assert "writes type 'plugin_art'" in _registration.failure_note(plugins.ARTIFACTS, "plugin_art_legacy")


def test_load_failures_lists_every_plugin_that_did_not_take_effect(monkeypatch):
    broken = _EP("broken_box", "_RecordingProvider")
    broken.value = "agentenv_broken.nowhere:Provider"
    _install(
        monkeypatch,
        envs=[_EP("browser", "_BrowserEnv"), _EP("mcp_server", "_BrowserEnv"), _EP("web_browser", "_BrowserEnv")],
        sandbox_providers=[broken, _EP("plugin_box", "_RecordingProvider")],
    )
    config = get_config()
    config.env_registry()
    config.sandbox_registry()

    failures = plugins.load_failures()

    assert set(failures) == {plugins.ENVS, plugins.SANDBOX_PROVIDERS}
    assert set(failures[plugins.ENVS]) == {"mcp_server", "web_browser"}
    assert "clashes with a built-in" in failures[plugins.ENVS]["mcp_server"]
    assert set(failures[plugins.SANDBOX_PROVIDERS]) == {"broken_box"}


def test_load_failures_is_empty_when_every_plugin_took_effect(monkeypatch):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    get_config().env_registry()

    assert plugins.load_failures() == {}


def test_a_new_config_starts_a_new_failure_record(monkeypatch):
    broken = _EP("broken_box", "_RecordingProvider")
    broken.value = "agentenv_broken.nowhere:Provider"
    _install(monkeypatch, sandbox_providers=[broken])
    old = get_config()
    old.sandbox_registry()
    assert set(plugins.load_failures()[plugins.SANDBOX_PROVIDERS]) == {"broken_box"}

    reset_config()

    assert plugins.load_failures() == {}
    assert _registration.failure_note(plugins.SANDBOX_PROVIDERS, "broken_box") == ""
    assert set(plugins.load_failures(old)[plugins.SANDBOX_PROVIDERS]) == {"broken_box"}


@pytest.mark.parametrize("reset", [True, False], ids=["after-reset", "same-config"])
def test_a_registry_built_while_a_plugin_is_importing_is_not_kept(monkeypatch, reset):
    _install(monkeypatch, envs=[_ReentrantEP("browser", "_BrowserEnv", reset=reset)])

    get_config().env_registry()

    assert get_config().env_registry()["browser"] is _BrowserEnv
    assert plugins.load_failures() == {}


class _BuildsAnotherRegistryWhenReentered(_EP):
    """A plugin whose import re-enters its own registry, and whose re-import builds another."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def load(self):
        self.calls += 1
        if self.calls == 1:
            get_config().env_registry()
        elif self.calls == 2:
            get_config().task_step_registry()
        return super().load()


def test_a_registry_first_built_inside_a_reentered_build_records_its_failures(monkeypatch):
    broken = _EP("broken_step", "_PluginStep")
    broken.value = "agentenv_broken.nowhere:Missing"
    _install(monkeypatch, envs=[_BuildsAnotherRegistryWhenReentered("browser", "_BrowserEnv")], task_steps=[broken])

    get_config().env_registry()

    assert list(plugins.load_failures()) == [plugins.TASK_STEPS]
    assert list(plugins.load_failures()[plugins.TASK_STEPS]) == ["broken_step"]


def test_a_real_failure_beside_a_reentrant_plugin_is_still_recorded(monkeypatch):
    # The re-entered build records nothing; the outer build must still record both broken plugins,
    # one sorting before the re-entrant plugin and one after it.
    broken = []
    for name in ("a_broken", "z_broken"):
        ep = _EP(name, "_BrowserEnv")
        ep.value = "agentenv_broken.nowhere:Missing"
        broken.append(ep)
    _install(monkeypatch, envs=[broken[0], _ReentrantEP("browser", "_BrowserEnv", reset=False), broken[1]])

    get_config().env_registry()

    assert set(plugins.load_failures()[plugins.ENVS]) == {"a_broken", "z_broken"}


def test_a_plugin_that_exits_while_imported_is_a_load_failure_not_fatal(monkeypatch):
    def exit_on_import():
        raise SystemExit(3)

    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv", on_load=exit_on_import)])

    registry = get_config().env_registry()

    assert "browser" not in registry
    assert "SystemExit(3)" in plugins.load_failures()[plugins.ENVS]["browser"]


def test_a_subclass_inheriting_a_built_in_type_is_told_to_define_its_own(monkeypatch):
    _install(monkeypatch, envs=[_EP("my_website", "_MyWebsite")], artifacts=[_EP("my_file", "_MyFile")])
    config = get_config()
    config.env_registry()
    config.artifact_registry()

    assert "inherits the built-in type 'website'" in _registration.failure_note(plugins.ENVS, "my_website")
    assert "inherits the built-in type 'file'" in _registration.failure_note(plugins.ARTIFACTS, "my_file")


def test_an_artifact_extra_name_its_type_field_rejects_is_skipped(monkeypatch):
    """Registered, it would read nothing: every document stored under it fails validation."""
    _install(monkeypatch, artifacts=[
        _EP("strict_art", "_StrictArtifact"),
        _EP("strict_art_legacy", "_StrictArtifact"),
    ])

    registry = get_config().artifact_registry()

    assert registry["strict_art"] is _StrictArtifact and "strict_art_legacy" not in registry
    assert "does not accept 'strict_art_legacy'" in _registration.failure_note(plugins.ARTIFACTS, "strict_art_legacy")


def test_an_explorer_plugin_that_fails_to_construct_is_skipped(monkeypatch):
    _install(monkeypatch, explorer_plugins=[_EP("broken_routes", "_BrokenCtorRoutes"), _EP("plugin_routes", "_Routes")])

    assert [type(p) for p in load_plugins()] == [_Routes]
    reason = plugins.load_failures()[plugins.EXPLORER_PLUGINS]["broken_routes"]
    assert "failed to construct: RuntimeError('boom in ctor')" in reason


def test_an_explorer_plugin_router_is_read_once_when_the_app_mounts_it(monkeypatch):
    _install(monkeypatch, explorer_plugins=[_EP("counting_routes", "_CountingRoutes")])
    monkeypatch.setattr(_CountingRoutes, "reads", 0)

    load_plugins()
    assert _CountingRoutes.reads == 0

    create_app()
    assert _CountingRoutes.reads == 1


def test_config_explorer_plugins_are_still_constructed_in_list_order(monkeypatch, tmp_path):
    """The first impl's own failure surfaces, not a later impl's bad pointer, as before plugins."""
    _use_config(monkeypatch, tmp_path, f"""
        [explorer.plugins]
        impls = ["{_HERE}:_BrokenCtorRoutes", "agentenv_missing.nowhere:Routes"]
    """)

    with pytest.raises(RuntimeError, match="boom in ctor"):
        load_plugins()


# ---------------------------------------------------------------- config over plugins


def test_config_impl_replaces_a_plugin_with_a_warning(monkeypatch, tmp_path, warnings_logged):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    _use_config(monkeypatch, tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_OtherBrowserEnv"]
    """)

    assert get_config().env_registry()["browser"] is _OtherBrowserEnv
    assert any("replaces the class registered by plugin 'browser'" in m for m in warnings_logged())


def test_config_impl_naming_the_plugins_own_class_is_silent(monkeypatch, tmp_path, warnings_logged):
    """The transitional sdk declares every class twice: as an entry point and in impls."""
    _install(
        monkeypatch,
        envs=[_EP("browser", "_BrowserEnv")],
        task_steps=[_EP("plugin_step", "_PluginStep")],
        artifacts=[_EP("plugin_art", "_PluginArtifact")],
    )
    _use_config(monkeypatch, tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_BrowserEnv"]
        [task_steps]
        impls = ["{_HERE}:_PluginStep"]
        [artifacts]
        impls = ["{_HERE}:_PluginArtifact"]
    """)
    config = get_config()

    assert config.env_registry()["browser"] is _BrowserEnv
    assert config.task_step_registry()["plugin_step"] is _PluginStep
    assert config.artifact_registry()["plugin_art"] is _PluginArtifact
    assert warnings_logged() == []


def test_two_config_impls_for_one_plugin_name_still_collide(monkeypatch, tmp_path):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    _use_config(monkeypatch, tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_OtherBrowserEnv", "{_HERE}:_BrowserEnv"]
    """)

    with pytest.raises(ConfigError, match="already registered"):
        get_config().env_registry()


def test_config_explorer_plugin_replaces_a_plugin_in_place(monkeypatch, tmp_path, warnings_logged):
    _install(monkeypatch, explorer_plugins=[_EP("plugin_routes", "_Routes")])
    _use_config(monkeypatch, tmp_path, f"""
        [explorer.plugins]
        impls = ["{_HERE}:_OtherRoutes"]
    """)

    assert [type(p) for p in load_plugins()] == [_OtherRoutes]
    assert any("replaces the class registered by plugin" in m for m in warnings_logged())


def test_config_replacing_an_artifact_class_strands_none_of_its_names(monkeypatch, tmp_path):
    _install(monkeypatch, artifacts=[
        _EP("plugin_art", "_PluginArtifact"),
        _EP("plugin_art_legacy", "_PluginArtifact"),
    ])
    _use_config(monkeypatch, tmp_path, f"""
        [artifacts]
        impls = ["{_HERE}:_OtherPluginArtifact"]
    """)

    registry = get_config().artifact_registry()

    assert registry["plugin_art"] is _OtherPluginArtifact
    assert "plugin_art_legacy" not in registry
    assert "resolves to _OtherPluginArtifact" in _registration.failure_note(plugins.ARTIFACTS, "plugin_art_legacy")


def test_an_alias_to_a_plugin_that_failed_to_load_leaves_artifact_reads_working(
    monkeypatch, tmp_path, warnings_logged
):
    broken = _EP("plugin_art", "_PluginArtifact")
    broken.value = "agentenv_broken.nowhere:Artifact"
    _install(monkeypatch, artifacts=[broken])
    _use_config(monkeypatch, tmp_path, """
        [artifacts.type_aliases]
        plugin_art_old = "plugin_art"
    """)

    registry = get_config().artifact_registry()

    assert registry["file"] is FileArtifact
    assert any("whose plugin failed to load; skipped" in m for m in warnings_logged())
    assert "ModuleNotFoundError" in _registration.failure_note(plugins.ARTIFACTS, "plugin_art")


def test_an_alias_colliding_with_a_plugin_name_names_the_plugin(monkeypatch, tmp_path):
    _install(monkeypatch, artifacts=[_EP("plugin_art", "_PluginArtifact")])
    _use_config(monkeypatch, tmp_path, """
        [artifacts.type_aliases]
        plugin_art = "file"
    """)

    with pytest.raises(ConfigError, match=r"but plugin 'plugin_art' from agentenv-demo 1\.0"):
        get_config().artifact_registry()


def _conflicted(monkeypatch):
    _install(
        monkeypatch,
        envs=[_EP("browser", "_BrowserEnv", dist="a"), _EP("browser", "_OtherBrowserEnv", dist="b")],
        artifacts=[_EP("plugin_art", "_PluginArtifact", dist="a"), _EP("plugin_art_legacy", "_PluginArtifact", dist="a"),
                   _EP("plugin_art", "_OtherPluginArtifact", dist="b")],
        sandbox_providers=[_EP("plugin_box", "_RecordingProvider", dist="a"),
                           _EP("plugin_box", "_OtherRecordingProvider", dist="b")],
        state_providers=[_EP("plugin_state", "_PluginState", dist="a"), _EP("plugin_state", "_OtherPluginState", dist="b")],
        explorer_plugins=[_EP("plugin_routes", "_Routes", dist="a"), _EP("plugin_routes", "_OtherRoutes", dist="b")],
    )


def test_config_cannot_settle_a_conflict(monkeypatch, tmp_path, warnings_logged):
    _conflicted(monkeypatch)
    _use_config(monkeypatch, tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_BrowserEnv"]
        [artifacts]
        impls = ["{_HERE}:_PluginArtifact"]
        [artifacts.type_aliases]
        plugin_art_old = "plugin_art"
        [sandbox.providers.plugin_box]
        impl = "{_HERE}:_RecordingProvider"
        [state.providers.plugin_state.config]
        secret_name = "demo"
        [explorer.plugins]
        impls = ["{_HERE}:_Routes"]
    """)
    config = get_config()

    assert "browser" not in config.env_registry() and "plugin_art" not in config.artifact_registry()
    assert "plugin_box" not in config.sandbox_registry() and "plugin_state" not in config.state_registry()
    assert load_plugins() == []
    skipped = [m for m in warnings_logged() if " and was skipped: " in m]
    for where in ("[envs] impl", "[artifacts] impl", "[artifacts] type_aliases 'plugin_art_old'",
                  "[sandbox.providers.plugin_box]", "[state.providers.plugin_state]", "[explorer.plugins] impl"):
        assert any(m.startswith(where) and "Remove all but one" in m for m in skipped), where
    with pytest.raises(ValueError, match="Unknown sandbox backend: 'plugin_box' .*2 installed plugins register"):
        build_sandbox_provider("plugin_box")


def test_an_alias_from_a_conflicted_name_names_both_plugins(monkeypatch, tmp_path):
    _conflicted(monkeypatch)
    _use_config(monkeypatch, tmp_path, """
        [artifacts.type_aliases]
        plugin_art = "file"
    """)

    with pytest.raises(ConfigError, match=r"but plugins 'plugin_art' from a 1\.0 .*; 'plugin_art' from b 1\.0"):
        get_config().artifact_registry()


@pytest.mark.parametrize(("body", "build", "match"), [
    ('[sandbox.providers.plugin_box]\nimpl = "agentenv_nowhere.providers:Missing"\n', "sandbox_registry", "Cannot import impl"),
    ("[sandbox.providers]\nplugin_box = 5\n", "sandbox_registry", ""),
    (f'[state.providers.plugin_state]\nimpl = "{_HERE}:_SiblingState"\n', "state_registry", "they must match"),
])
def test_a_provider_table_for_a_conflicted_name_is_still_checked(monkeypatch, tmp_path, body, build, match):
    _conflicted(monkeypatch)
    _use_config(monkeypatch, tmp_path, body)

    with pytest.raises((ConfigError, TypeError), match=match):
        getattr(get_config(), build)()


def test_a_stored_artifact_of_a_conflicted_type_names_the_claimants_not_config(monkeypatch):
    _conflicted(monkeypatch)

    with pytest.raises(ValueError) as raised:
        ArtifactStore()._deserialize({"id": "a", "version": 1, "type": "plugin_art"})

    assert "2 installed plugins register 'plugin_art'" in str(raised.value)
    assert "[artifacts].impls" not in str(raised.value)


def test_an_extra_artifact_name_of_a_conflicted_type_says_why_it_is_left_out(monkeypatch):
    _conflicted(monkeypatch)

    assert "plugin_art_legacy" not in get_config().artifact_registry()
    note = _registration.failure_note(plugins.ARTIFACTS, "plugin_art_legacy")
    assert "its class writes type 'plugin_art', which is left out: 2 installed plugins register" in note
    assert "Remove all but one" in note


# ---------------------------------------------------------------- the provider registries


def test_sandbox_transitional_shape_resolves_without_a_collision(monkeypatch, tmp_path, warnings_logged):
    """Entry point plus a surviving impl table naming the same class, and it is the default."""
    _install(monkeypatch, sandbox_providers=[_EP("plugin_box", "_RecordingProvider")])
    _use_config(monkeypatch, tmp_path, f"""
        [sandbox]
        default = "plugin_box"
        [sandbox.providers.plugin_box]
        impl = "{_HERE}:_RecordingProvider"
        [sandbox.providers.plugin_box.config]
        region = "us-west-2"
    """)

    provider = get_sandbox_provider()

    assert isinstance(provider, _RecordingProvider) and provider.config == {"region": "us-west-2"}
    assert warnings_logged() == []


def test_sandbox_config_only_table_configures_a_plugin(monkeypatch, tmp_path):
    _install(monkeypatch, sandbox_providers=[_EP("plugin_box", "_RecordingProvider")])
    _use_config(monkeypatch, tmp_path, """
        [sandbox.providers.plugin_box.config]
        region = "us-west-2"
    """)

    assert build_sandbox_provider("plugin_box").config == {"region": "us-west-2"}


def test_sandbox_config_impl_replaces_a_plugin_with_a_warning(monkeypatch, tmp_path, warnings_logged):
    _install(monkeypatch, sandbox_providers=[_EP("plugin_box", "_RecordingProvider")])
    _use_config(monkeypatch, tmp_path, f"""
        [sandbox.providers.plugin_box]
        impl = "{_HERE}:_OtherRecordingProvider"
    """)

    assert type(build_sandbox_provider("plugin_box")) is _OtherRecordingProvider
    assert any("[sandbox.providers.plugin_box] impl" in m and "replaces" in m for m in warnings_logged())


def test_sandbox_config_for_a_plugin_that_failed_to_load_does_not_take_the_registry_down(
    monkeypatch, tmp_path
):
    broken = _EP("broken_box", "_RecordingProvider")
    broken.value = "agentenv_broken.nowhere:Provider"
    _install(monkeypatch, sandbox_providers=[broken])
    _use_config(monkeypatch, tmp_path, """
        [sandbox.providers.broken_box.config]
        region = "us-west-2"
    """)

    assert build_sandbox_provider("local")
    with pytest.raises(ValueError, match="failed to load"):
        build_sandbox_provider("broken_box")


def test_plugin_sandbox_providers_keep_the_deploy_time_type_guard(monkeypatch, tmp_path):
    _install(monkeypatch, sandbox_providers=[_EP("plugin_box", "_GuardedProvider")])
    _use_config(monkeypatch, tmp_path, """
        [sandbox.providers.plugin_box.config]
        produced_type = "something_else"
    """)
    provider = build_sandbox_provider("plugin_box")

    with pytest.raises(SandboxProviderTypeError):
        asyncio.run(provider.create_sandbox())
    assert provider.last_sandbox.terminated


def test_state_transitional_shape_and_config_only_table(monkeypatch, tmp_path, warnings_logged):
    _install(monkeypatch, state_providers=[_EP("plugin_state", "_PluginState")])
    _use_config(monkeypatch, tmp_path, f"""
        [state.providers.plugin_state]
        impl = "{_HERE}:_PluginState"
        [state.providers.plugin_state.config]
        secret_name = "dev/bundle"
    """)
    assert build_state_provider("plugin_state").config == {"secret_name": "dev/bundle"}

    _use_config(monkeypatch, tmp_path, """
        [state.providers.plugin_state.config]
        secret_name = "prod/bundle"
    """)
    assert build_state_provider("plugin_state").config == {"secret_name": "prod/bundle"}
    assert warnings_logged() == []


def test_state_config_impl_replaces_a_plugin_with_a_warning(monkeypatch, tmp_path, warnings_logged):
    _install(monkeypatch, state_providers=[_EP("plugin_state", "_PluginState")])
    _use_config(monkeypatch, tmp_path, f"""
        [state.providers.plugin_state]
        impl = "{_HERE}:_OtherPluginState"
    """)

    assert type(build_state_provider("plugin_state")) is _OtherPluginState
    assert any("[state.providers.plugin_state] impl" in m and "replaces" in m for m in warnings_logged())


def test_loading_a_plugin_cannot_change_which_config_a_config_reads(monkeypatch, tmp_path_factory):
    """A plugin package may point AGENT_ENV_CONFIG at its own file on import, as a platform
    bootstrap does. The Config loading it has already read its document and keeps it."""
    bare = tmp_path_factory.mktemp("bare")
    elsewhere = bare / "elsewhere.toml"
    elsewhere.write_text('[sandbox]\ndefault = "somewhere_else"\n')
    monkeypatch.chdir(bare)
    monkeypatch.delenv("AGENT_ENV_CONFIG")
    reset_config()
    _install(monkeypatch, envs=[
        _EP("browser", "_BrowserEnv", on_load=lambda: monkeypatch.setenv("AGENT_ENV_CONFIG", str(elsewhere)))
    ])
    config = get_config()

    assert config.env_registry()["browser"] is _BrowserEnv
    assert config.config_path() is None
    assert config.section("sandbox") == {}
    # The side effect still happened; the next Config built in this process reads it.
    reset_config()
    assert get_config().config_path() == elsewhere


def test_load_impl_guards_an_already_loaded_class():
    assert load_impl(_RecordingProvider, SandboxProvider) is _RecordingProvider
    with pytest.raises(ConfigError, match="not a subclass of SandboxProvider"):
        load_impl(_BrowserEnv, SandboxProvider)


# ---------------------------------------------------------------- a real installed distribution


def test_discovery_imports_nothing_until_the_registry_is_built(tmp_path, monkeypatch):
    (tmp_path / "agentenv_real_demo.py").write_text(
        "from agent_env.env.env import Env\n\n\nclass RealEnv(Env):\n    type = 'real_demo'\n\n"
        "    @classmethod\n    def from_dict(cls, data):\n        return cls(data['id'], data.get('version'))\n"
    )
    dist_info = tmp_path / "agentenv_real_demo-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: agentenv-real-demo\nVersion: 0.1.0\n")
    (dist_info / "entry_points.txt").write_text("[agent_env.envs]\nreal_demo = agentenv_real_demo:RealEnv\n")
    (dist_info / "RECORD").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    try:
        discovered = _discovery.discover(plugins.ENVS)

        # Only this test's distribution: plugins already installed in the venv must not matter.
        assert [(p.name, p.dist, p.version) for p, _ in discovered["real_demo"]] == [
            ("real_demo", "agentenv-real-demo", "0.1.0")
        ]
        assert "agentenv_real_demo" not in sys.modules
        assert get_config().env_registry()["real_demo"].__module__ == "agentenv_real_demo"
    finally:
        sys.modules.pop("agentenv_real_demo", None)
