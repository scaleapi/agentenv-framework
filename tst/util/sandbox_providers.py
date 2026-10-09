"""A sandbox provider for unit tests that stand one in with a MagicMock: it declares what a real provider declares, that
it is its own only link, and that its sandboxes reach a URL as it is."""

from unittest.mock import MagicMock

from agent_env.providers.sandbox_providers.sandbox_provider import Accepts


def mock_provider(*, on_this_machine: bool = False, **attrs) -> MagicMock:
    """A MagicMock provider that declares what a VM provider does, unless ``attrs`` declare otherwise."""
    facts = {"CREATES_VMS": True, "SANDBOX_ACCEPTS": Accepts.LOADABLE, "CONTAINER_ACCEPTS": Accepts.NAME,
             "PRIVATE_NETWORK": False}
    provider = MagicMock(ON_THIS_MACHINE=on_this_machine, **(facts | attrs))
    provider.links = (provider,)
    provider.url_from_sandbox.side_effect = lambda url: url
    return provider
