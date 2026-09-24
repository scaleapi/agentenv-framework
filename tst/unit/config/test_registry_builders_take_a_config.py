"""Every registry builder can be handed the Config to build from.

They each read the ambient document for themselves, which is fine while one process has one
document. It stops being fine once a `Config` owns its document: the Config has to be able to
say "build from *this*", or its registries would resolve independently of it and the object
would not, in fact, hold the config. Every in-repo caller now passes it; the optional
default remains for direct callers outside this package.
"""

from __future__ import annotations

import pytest

from agent_env.artifact.registry import _merge_config_toml_artifacts
from agent_env.config import snapshot as config_snapshot

from tst.util.config import config_with_document
from agent_env.config.errors import ConfigError
from agent_env.env.registry import _merge_config_toml_envs
from agent_env.explorer.plugin import load_plugins
from agent_env.providers.sandbox_provider import _merge_config_toml_sandbox_providers
from agent_env.providers.state.env_state_provider import _merge_config_toml_state_providers
from agent_env.task_step.registry import _merge_config_toml_steps

# Each builder, the top-level section it reads, and a call that builds into a throwaway
# registry. `load_plugins` is the odd one out: it is public and returns its result.
_BUILDERS = [
    ("envs", lambda **kw: _merge_config_toml_envs({}, **kw)),
    ("artifacts", lambda **kw: _merge_config_toml_artifacts({}, **kw)),
    ("task_steps", lambda **kw: _merge_config_toml_steps({}, **kw)),
    ("sandbox", lambda **kw: _merge_config_toml_sandbox_providers({}, **kw)),
    ("state", lambda **kw: _merge_config_toml_state_providers({}, **kw)),
    ("explorer", lambda **kw: load_plugins(**kw)),
]
_IDS = [name for name, _ in _BUILDERS]


@pytest.mark.parametrize("section, build", _BUILDERS, ids=_IDS)
def test_a_builder_reads_the_document_it_is_given(tmp_path, monkeypatch, section, build):
    """Proved through the shape rule rather than a registration: a mis-shaped section raises,
    so a builder that read the ambient document — which is well-formed here — would not."""
    path = tmp_path / "config.toml"
    path.write_text("")                       # ambient: empty, every section absent and legal
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    given = config_with_document({section: 5})

    with pytest.raises(ConfigError, match=rf"\[{section}\] must be a table"):
        build(source=given)


@pytest.mark.parametrize("section, build", _BUILDERS, ids=_IDS)
def test_a_given_document_is_not_re_discovered(tmp_path, monkeypatch, section, build):
    """The property a `Config` will depend on. Reading the passed document but discovering
    anyway would leave the Config's registries agreeing with it by luck — right answer, wrong
    reason, and wrong the moment the two disagree."""
    def no_discovery(*_a, **_k):
        raise AssertionError("discovery ran even though a document was supplied")

    monkeypatch.setattr(config_snapshot.loader, "discover_config_path", no_discovery)

    build(source=config_with_document({}))


@pytest.mark.parametrize("section, build", _BUILDERS, ids=_IDS)
def test_omitting_it_still_reads_the_ambient_document(tmp_path, monkeypatch, section, build):
    """The compatibility half. Plugin call sites pass a registry and
    nothing else, and rely on these reading whatever `AGENT_ENV_CONFIG` points at now."""
    path = tmp_path / "config.toml"
    path.write_text(f"{section} = 5\n")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))

    with pytest.raises(ConfigError, match=rf"\[{section}\] must be a table"):
        build()
