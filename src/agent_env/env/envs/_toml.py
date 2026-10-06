"""What an env type written from an env.toml checks first: its keys, its env_provider_type and, for a type that
names itself, its environment_name."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_env.providers.env_providers.constants import GATEWAY_SERVICE_NAMES

if TYPE_CHECKING:
    from agent_env.bundle.authoring import AuthoringContext


def accept_env_toml(cls: type, data: dict[str, Any], ctx: AuthoringContext, *, image_key: str | None = None,
                    one_server: bool = False) -> tuple[dict[str, Any], list[str]]:
    """The keys of ``data`` ``cls`` takes, and the problems with them. With ``image_key``, the env is named by
    ``environment_name`` or, left out, by the ``@environment_card`` in the source of the image that key builds;
    ``one_server`` lets env_provider_type name a provider that deploys one MCP server."""
    # The env types import before the providers that deploy them, which import those types.
    from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
    from agent_env.providers.env_providers.env_provider import _env_provider_class
    from agent_env.providers.env_providers.env_server_provider import EnvironmentServerProvider

    fields, problems = ctx.accepted(data, **cls.toml_keys)
    provider_type = fields.get("env_provider_type", "gateway")
    try:
        provider = _env_provider_class(provider_type)
    except ValueError as e:
        problems.append(f"env.toml: env_provider_type: {e}")
        provider = None
    if provider is not None and not one_server and issubclass(provider, EnvironmentServerProvider):
        problems.append(f"env.toml: env_provider_type {provider_type!r} deploys one MCP server, not a {cls.type} env")
    if image_key is not None:
        name = _environment_name(data, fields, ctx, image_key, problems)
        if name in GATEWAY_SERVICE_NAMES and provider is not None and issubclass(provider, EnvironmentGatewayProvider):
            problems.append(f"env.toml: environment_name {name!r} is one a gateway deploy names its own containers "
                            f"({', '.join(sorted(GATEWAY_SERVICE_NAMES))}); choose another")
    return fields, problems


def _environment_name(data: dict[str, Any], fields: dict[str, Any], ctx: AuthoringContext, image_key: str,
                      problems: list[str]) -> str | None:
    if "environment_name" in data:
        if fields.get("environment_name") == "":
            problems.append("env.toml: environment_name can't be empty")
        return fields.get("environment_name")
    names = ctx.card_names(image_key)
    what = "image" if image_key == "image" else image_key.replace("_", " ")
    if names is None:
        problems.append(f"env.toml: environment_name isn't set, and its {what} isn't built from this folder, so "
                        "there's no source to read it from; set it")
    elif len(names) != 1:
        found = (f"several environment cards ({', '.join(map(repr, names))})" if names
                 else "no @environment_card(name=...)")
        problems.append(f"env.toml: environment_name isn't set, and the source its {what} is built from declares "
                        f"{found}; set it")
    else:
        fields["environment_name"] = names[0]
    return fields.get("environment_name")
