"""Artifact registry for AgentEnv.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter, ValidationError

from agent_env.plugins import _registration
from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.cli import CliArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.artifact.artifacts.environment_universe import EnvironmentUniverseArtifact
from agent_env.artifact.artifacts.skill import SkillArtifact
from agent_env.artifact.artifacts.vm_image import VMImageArtifact

if TYPE_CHECKING:
    from agent_env.config.runtime import Config

logger = logging.getLogger(__name__)


def _get_type(cls: type[Artifact]) -> Any:
    return cls.model_fields["type"].default


ARTIFACT_REGISTRY: dict[str, type[Artifact]] = {
    _get_type(CliArtifact): CliArtifact,
    _get_type(FileArtifact): FileArtifact,
    _get_type(FileArtifactUniverse): FileArtifactUniverse,
    _get_type(DockerImageArtifact): DockerImageArtifact,
    _get_type(EnvironmentArtifact): EnvironmentArtifact,
    _get_type(EnvironmentUniverseArtifact): EnvironmentUniverseArtifact,
    _get_type(SkillArtifact): SkillArtifact,
    _get_type(VMImageArtifact): VMImageArtifact,
}




def _build_registry(source: Config | None = None) -> dict[str, type[Artifact]]:
    """The built-in artifact types, then ``agent_env.artifacts`` plugins, then ``[artifacts] impls``
    in ``source``."""
    registry = dict(ARTIFACT_REGISTRY)
    from_plugins = _registration.merge(registry, _registration.ARTIFACTS, _validate_plugin, source=source)
    _merge_config_toml_artifacts(registry, source=source, from_plugins=from_plugins)
    # After config, which may have replaced a class's own spelling and stranded its extra names.
    for name in from_plugins:
        if reason := _extra_name_problem(name, registry[name], registry):
            from_plugins.reject(name, reason)
    # This document's aliases, not the ambient Config's, or the two can disagree.
    _check_type_aliases(registry, _load_type_aliases(source), from_plugins=from_plugins)
    return registry


def _extra_name_problem(name: str, cls: type[Artifact], registry: dict[str, type[Artifact]]) -> str | None:
    """Why a plugin's extra name (a legacy spelling) for ``cls`` would read nothing, if it would.

    It only reads old documents: the class writes its own spelling, which must resolve to it,
    and a stored document validates against the class's ``type`` field, which must accept it.
    """
    own = _get_type(cls)
    if name == own:
        return None
    builtin = ARTIFACT_REGISTRY.get(own)
    if builtin is not None and builtin is not cls:
        return f"its class inherits the built-in type {own!r}; give it its own 'type' field default"
    if registry.get(own) is not cls:
        elsewhere = registry.get(own)
        where = "nothing registers it" if elsewhere is None else f"it resolves to {elsewhere.__qualname__}"
        return f"its class writes type {own!r}, but {where}; register that name to the class too"
    try:
        TypeAdapter(cls.model_fields["type"].annotation).validate_python(name)
    except ValidationError:
        return f"its 'type' field does not accept {name!r}, so no document stored under it can be read"
    return None


def _validate_plugin(name: str, loaded: Any) -> type[Artifact]:
    # One class may register under several names; _extra_name_problem checks the extra ones.
    cls = _registration.require_subclass(loaded, Artifact)
    if not isinstance(_get_type(cls), str):
        raise TypeError(f"{cls.__qualname__} does not define its own 'type' field default")
    return cls


def get_artifact_registry() -> dict[str, type[Artifact]]:
    from agent_env.config import runtime

    return runtime.get_config().artifact_registry()


def get_type_aliases() -> dict[str, str]:
    """Legacy artifact `type` spellings mapped to the canonical one, from config.

    Empty unless `[artifacts] type_aliases` declares any, and the core ships none: a
    deployment that renamed an artifact type keeps reading the documents it already
    wrote under the old spelling, without ever rewriting them. An installation that
    never renamed a type has nothing to declare.

    Registering a subclass under the old spelling expresses the same intent and stays
    supported; the two are alternatives rather than layers, so declaring both for one
    type is rejected.
    """
    from agent_env.config import runtime

    return runtime.get_config().artifact_type_aliases()


def canonical_type(stored_type: str) -> str:
    """The registry key `stored_type` resolves to — itself unless it is an alias."""
    return get_type_aliases().get(stored_type, stored_type)


def equivalent_types(artifact_type: str) -> list[str]:
    """Every stored spelling that resolves to the same artifact as `artifact_type`.

    A `type` filter is evaluated server-side against the value as stored, so it has to
    match documents written under either spelling — canonicalising on read happens far
    too late to help it. Returns `[artifact_type]` unchanged when nothing aliases to it,
    so an unaliased type still compiles to a scalar equality rather than a one-element
    `$in`. The order is unspecified — `canonical_type` gives the representative spelling.
    """
    aliases = get_type_aliases()
    canonical = aliases.get(artifact_type, artifact_type)
    return sorted({artifact_type, canonical} | {a for a, c in aliases.items() if c == canonical})


def _load_type_aliases(source: Config | None = None) -> dict[str, str]:
    from agent_env.config import runtime
    from agent_env.config import ConfigError

    # A nested walk, so a mis-shaped table raises the config layer's own message rather
    # than a second hand-rolled one.
    aliases = (source or runtime.get_config()).section("artifacts", "type_aliases")
    for legacy, canonical in aliases.items():
        if not isinstance(canonical, str):
            raise ConfigError(
                f"[artifacts] type_aliases {legacy!r} must map to a string, got "
                f"{type(canonical).__name__}: {canonical!r}"
            )
        if canonical == legacy:
            raise ConfigError(f"[artifacts] type_aliases {legacy!r} maps to itself")
        if canonical in aliases:
            raise ConfigError(
                f"[artifacts] type_aliases {legacy!r} maps to {canonical!r}, which is itself "
                f"an alias; chains are not resolved"
            )
    return dict(aliases)


def _check_type_aliases(
    registry: dict[str, type[Artifact]],
    aliases: dict[str, str],
    *,
    from_plugins: _registration.Registrations | None = None,
) -> None:
    from agent_env.config import ConfigError

    from_plugins = from_plugins if from_plugins is not None else _registration.Registrations.empty()
    for legacy, canonical in aliases.items():
        if canonical not in registry:
            if from_plugins.failed(canonical):
                # Like a config-only provider table for a failed plugin: skipped, not fatal, so
                # one broken plugin does not take every artifact read down with it.
                logger.warning(
                    "[artifacts] type_aliases %r maps to %r, whose plugin failed to load; skipped", legacy, canonical
                )
                continue
            raise ConfigError(
                f"[artifacts] type_aliases {legacy!r} maps to {canonical!r}, which no artifact "
                f"registers; declare it under [artifacts].impls or correct the alias"
            )
        # Aliasing a type that is still registered is normal mid-rename, as long as both
        # spellings mean the same class. Pointing it at a different class instead silently
        # redirects every document that type owns.
        owner = registry.get(legacy)
        if owner is not None and owner is not registry[canonical]:
            if (plugin := from_plugins.plugin(legacy)) is not None:
                raise ConfigError(
                    f"[artifacts] type_aliases {legacy!r} maps to {canonical!r}, but plugin {plugin} "
                    f"registers {legacy!r} to {owner.__name__}; uninstall the plugin or drop the alias"
                )
            raise ConfigError(
                f"[artifacts] type_aliases {legacy!r} is registered to {owner.__name__} but "
                f"aliased to {canonical!r} ({registry[canonical].__name__}); a type resolves "
                f"either through its own registered class or through an alias, not both — "
                f"drop it from impls, or drop the alias"
            )


def _merge_config_toml_artifacts(
    registry: dict[str, type[Artifact]],
    *,
    source: Config | None = None,
    from_plugins: _registration.Registrations | None = None,
) -> None:
    from agent_env.config import runtime
    from agent_env.config import ConfigError, load_impl

    section = (source or runtime.get_config()).section("artifacts")
    impls = section.get("impls", [])
    if not isinstance(impls, list):
        raise ConfigError(
            f"[artifacts] impls must be a list of 'module:Class' strings, got {type(impls).__name__}"
        )
    from_plugins = from_plugins if from_plugins is not None else _registration.Registrations.empty()
    for impl in impls:
        if not isinstance(impl, str):
            raise ConfigError(
                f"[artifacts] impl must be a 'module:Class' string, got {type(impl).__name__}: {impl!r}"
            )
        cls = load_impl(impl, Artifact)
        artifact_type = _get_type(cls)
        if not isinstance(artifact_type, str):
            raise ConfigError(
                f"[artifacts] impl {impl!r} does not define its own 'type' "
                f"(inherits the base Artifact default); set a concrete 'type' field default"
            )
        if not from_plugins.release(artifact_type, f"[artifacts] impl {impl!r}", cls) and artifact_type in registry:
            raise ConfigError(
                f"[artifacts] impl {impl!r} type {artifact_type!r} is already registered "
                f"(conflicts with a built-in or another custom artifact)"
            )
        registry[artifact_type] = cls
