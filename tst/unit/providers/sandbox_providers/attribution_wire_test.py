"""Byte-identity net for the emitted compute-attribution wire.

Attribution is an open dict: every key lands on the Modal sandbox `tags`, and the shared Modal
App's `tags` carry only the configured defaults (the `labels` sink of the out-of-tree platform
sandbox provider is pinned by that provider's tests). These pin what comes out, so a refactor of
how attribution is carried announces itself instead of quietly re-bucketing spend.

Precedence is `explicit arg -> config.toml [sandbox.attribution]`, tested `is not None`: an
empty string is a value.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.attribution import RUN_ID_KEY
from agent_env.config import get_config
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider, _attribution_tags
from tst.unit.store.fakes import FakeImageStore

_DEFAULTS = {"org": "o0", "team": "t0"}


def _use_config(tmp_path, monkeypatch, attribution: dict[str, str]) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("[sandbox.attribution]\n" + "".join(f'{k} = "{v}"\n' for k, v in attribution.items()))
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))


@pytest.fixture(autouse=True)
def _configured_defaults(tmp_path, monkeypatch):
    # Pin config discovery to a fixture so no ambient .agentenv leaks into the assertions.
    _use_config(tmp_path, monkeypatch, _DEFAULTS)


def _tags(**kwargs):
    return _attribution_tags(kwargs)


# --- the emitted dicts, exactly -------------------------------------------------


def test_modal_tags_with_nothing_supplied():
    assert _tags() == {"org": "o0", "team": "t0"}


def test_modal_tags_carry_any_key():
    assert _tags(team="t", cost_center="research", run_id="inst-1") == {
        "org": "o0", "team": "t", "cost_center": "research", "run_id": "inst-1",
    }


def test_modal_vm_shares_the_modal_sink():
    from agent_env.providers.sandbox_providers import modal_vm_sandbox

    assert modal_vm_sandbox._attribution_tags is _attribution_tags



# --- precedence -----------------------------------------------------------------


@pytest.mark.parametrize("field", list(_DEFAULTS))
def test_explicit_beats_the_configured_default(field):
    assert _tags(**{field: "explicit"})[field] == "explicit"


def test_a_configured_default_may_be_an_env_reference(tmp_path, monkeypatch):
    _use_config(tmp_path, monkeypatch, {**_DEFAULTS, "org": "env:ATTRIBUTION_TEST_ORG?o0"})
    monkeypatch.delenv("ATTRIBUTION_TEST_ORG", raising=False)
    assert _tags()["org"] == "o0"
    monkeypatch.setenv("ATTRIBUTION_TEST_ORG", "from-env")
    assert _tags()["org"] == "from-env"


def test_without_configured_defaults_modal_omits_unset_dimensions(tmp_path, monkeypatch):
    _use_config(tmp_path, monkeypatch, {})
    assert _tags() == {}
    assert _tags(team="t", unset=None) == {"team": "t"}


def test_an_empty_string_is_a_value_not_an_absence():
    """`is not None`, not truthiness — pinned because flipping it is silent on Modal."""
    assert _tags(team="")["team"] == ""


@pytest.mark.parametrize("key", ["", "cost center", "team/sub", "k" * 64])
def test_a_key_modal_cannot_take_is_refused_not_rewritten(key):
    with pytest.raises(ValueError, match="can't be a Modal tag"):
        _tags(**{key: "v"})


# --- what reaches Modal -----------------------------------------------------------


def _provider():
    get_config().set_image_store(FakeImageStore())
    provider = ModalSandboxProvider(app_name="agent-env-test")
    provider._get_app = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    provider._get_client = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    return provider


@pytest.mark.asyncio
async def test_the_shared_app_gets_the_defaults_and_the_sandbox_gets_the_run():
    provider = _provider()
    attribution = {"team": "t", "cost_center": "research", RUN_ID_KEY: "inst-1"}
    with patch("agent_env.providers.sandbox_providers.modal_sandbox.modal.Sandbox._experimental_create") as create:
        create.aio = AsyncMock(side_effect=RuntimeError("stop"))
        with pytest.raises(RuntimeError):
            await provider.create_container(image_name="img:latest", port=8000, env={}, attribution=attribution)
    assert provider._get_app.call_args.args == ("agent-env-test", _DEFAULTS)
    assert create.aio.call_args.kwargs["tags"] == {**_DEFAULTS, **attribution}


@pytest.mark.asyncio
async def test_a_bad_key_fails_before_the_app_is_looked_up():
    provider = _provider()
    with pytest.raises(ValueError, match="'cost center'"):
        await provider.create_container(image_name="img:latest", port=8000, env={}, attribution={"cost center": "x"})
    provider._get_app.assert_not_awaited()


# --- the dict is the only carried form -------------------------------------------


@pytest.mark.asyncio
async def test_a_chain_forwards_the_dict_as_is():
    """ChainedSandboxProvider takes **kwargs; the dict reaches the backend untouched, open
    keys included."""
    from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider

    seen = {}

    class Provider:
        type = "modern"

        async def create_sandbox(self, *, attribution=None, **kwargs):
            seen["attribution"] = attribution
            seen["kwargs"] = kwargs
            return SimpleNamespace(type="modern", network_policy=None)

    attribution = {"team": "t"}
    await ChainedSandboxProvider([Provider()]).create_sandbox(attribution=attribution, cpu=2.0)
    assert seen == {"attribution": attribution, "kwargs": {"cpu": 2.0}}


@pytest.mark.asyncio
async def test_the_quartet_is_no_longer_a_keyword_on_any_backend():
    """A caller still on the pre-dict signature fails loud rather than billing unattributed."""
    from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider

    with pytest.raises(TypeError, match="team"):
        await LocalSandboxProvider().create_vm(team="t")
