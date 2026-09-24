"""Byte-identity net for the emitted compute-attribution wire.

Attribution lands in the Modal App `tags` dict and the Modal App *name* (the `labels` sink of
the out-of-tree platform sandbox provider is pinned by that provider's tests). These pin what comes out, so a refactor of how attribution
is carried announces itself instead of quietly re-bucketing spend.

Precedence is `explicit arg -> config.toml [sandbox.attribution]`, tested `is not None`: an
empty string is a value. Modal accepts an empty project and `_app_name_for_project`
collapses every project into one App, so that case is pinned.
"""

from types import SimpleNamespace

import pytest

from agent_env.providers.modal_sandbox import (
    DEFAULT_APP_NAME,
    _app_name_for_project,
    _build_cost_attribution_tags,
)

_OBJECT_ID = "0123456789abcdef01234567"
_DEFAULT_PROJECT_ID = "fedcba9876543210fedcba98"
_DEFAULTS = {"product": "p0", "customer": "c0", "team": "t0", "project_id": _DEFAULT_PROJECT_ID}


def _use_config(tmp_path, monkeypatch, attribution: dict[str, str]) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("[sandbox.attribution]\n" + "".join(f'{k} = "{v}"\n' for k, v in attribution.items()))
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))


@pytest.fixture(autouse=True)
def _configured_defaults(tmp_path, monkeypatch):
    # Pin config discovery to a fixture so no ambient .agentenv leaks into the assertions.
    _use_config(tmp_path, monkeypatch, _DEFAULTS)


def _tags(**kwargs):
    return _build_cost_attribution_tags(kwargs)


# --- the emitted dicts, exactly -------------------------------------------------


def test_modal_tags_with_nothing_supplied():
    assert _tags() == {
        "product": "p0",
        "customer": "c0",
        "team": "t0",
        "project_id": _DEFAULT_PROJECT_ID,
    }


def test_modal_tags_with_everything_supplied():
    assert _tags(product="p", customer="c", team="t", project_id=_OBJECT_ID) == {
        "product": "p", "customer": "c", "team": "t", "project_id": _OBJECT_ID,
    }


def test_modal_vm_shares_the_modal_sink():
    from agent_env.providers import modal_vm_sandbox

    assert modal_vm_sandbox._build_cost_attribution_tags is _build_cost_attribution_tags


def test_the_default_project_names_its_own_modal_app():
    app_name = _app_name_for_project(DEFAULT_APP_NAME, _tags().get("project_id"))
    assert app_name == f"{DEFAULT_APP_NAME}-{_DEFAULT_PROJECT_ID}"
    assert len(app_name) <= 64  # Modal's limit; longer names are silently truncated


# --- precedence -----------------------------------------------------------------


@pytest.mark.parametrize("field", ["product", "customer", "team", "project_id"])
def test_explicit_beats_the_configured_default(field):
    assert _tags(**{field: "explicit"})[field] == "explicit"


def test_a_configured_default_may_be_an_env_reference(tmp_path, monkeypatch):
    _use_config(tmp_path, monkeypatch, {**_DEFAULTS, "product": "env:ATTRIBUTION_TEST_PRODUCT?p0"})
    monkeypatch.delenv("ATTRIBUTION_TEST_PRODUCT", raising=False)
    assert _tags()["product"] == "p0"
    monkeypatch.setenv("ATTRIBUTION_TEST_PRODUCT", "from-env")
    assert _tags()["product"] == "from-env"


def test_without_configured_defaults_modal_omits_unset_dimensions(tmp_path, monkeypatch):
    _use_config(tmp_path, monkeypatch, {})
    assert _tags() == {}
    assert _tags(team="t") == {"team": "t"}
    assert _app_name_for_project(DEFAULT_APP_NAME, _tags().get("project_id")) == DEFAULT_APP_NAME


@pytest.mark.parametrize("field", ["product", "customer", "team"])
def test_an_empty_string_is_a_value_not_an_absence(field):
    """`is not None`, not truthiness — pinned because flipping it is silent on Modal."""
    assert _tags(**{field: ""})[field] == ""


# --- the dict is the only carried form -------------------------------------------


@pytest.mark.asyncio
async def test_a_chain_forwards_the_dict_as_is():
    """ChainedSandboxProvider takes **kwargs; the dict reaches the backend untouched, open
    keys included."""
    from agent_env.providers.chained_sandbox_provider import ChainedSandboxProvider

    seen = {}

    class Provider:
        type = "modern"

        async def create_sandbox(self, *, attribution=None, **kwargs):
            seen["attribution"] = attribution
            seen["kwargs"] = kwargs
            return SimpleNamespace(type="modern", network_policy=None)

    attribution = {"project_id": _OBJECT_ID, "cost_center": "x"}
    await ChainedSandboxProvider([Provider()]).create_sandbox(attribution=attribution, cpu=2.0)
    assert seen == {"attribution": attribution, "kwargs": {"cpu": 2.0}}


@pytest.mark.asyncio
async def test_the_quartet_is_no_longer_a_keyword_on_any_backend():
    """A caller still on the pre-dict signature fails loud rather than billing unattributed."""
    from agent_env.providers.local_sandbox import LocalSandboxProvider

    with pytest.raises(TypeError, match="project_id"):
        await LocalSandboxProvider().create_vm(project_id=_OBJECT_ID)
