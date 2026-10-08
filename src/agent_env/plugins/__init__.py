"""Entry-point plugins: register types from an installed package.

An installed distribution adds a class to a registry by declaring an entry point. The group
says what kind of thing it is, the entry-point name is the registry key, and the value is the
class::

    [project.entry-points."agent_env.envs"]
    browser = "agentenv_browser.env:BrowserEnv"

A plugin's own settings live in the config file under ``[plugins.<its distribution name>]``, and
``settings`` reads them.

This package's top level is the public surface: the group names, ``load_failures``, ``settings``,
and ``inventory`` with the types it returns; its underscored modules are internal. The full contract
is in PLUGINS.md ("Register types from an installed package", "Plugin settings"), and the codes the
inventory reports are in "Plugin report format": codes are stable, reason text is not.
"""

from agent_env.config.plugin_tables import settings
from agent_env.plugins._inventory import (
    Claimant,
    Contribution,
    Diagnostic,
    Distribution,
    Inventory,
    Replacement,
    Status,
    inventory,
)
from agent_env.plugins._registration import (
    ARTIFACTS,
    ENV_PROVIDERS,
    ENVS,
    EXPLORER_PLUGINS,
    SANDBOX_PROVIDERS,
    STATE_PROVIDERS,
    TASK_STEPS,
    load_failures,
)

__all__ = [
    "ARTIFACTS",
    "ENV_PROVIDERS",
    "ENVS",
    "EXPLORER_PLUGINS",
    "SANDBOX_PROVIDERS",
    "STATE_PROVIDERS",
    "TASK_STEPS",
    "Claimant",
    "Contribution",
    "Diagnostic",
    "Distribution",
    "Inventory",
    "Replacement",
    "Status",
    "inventory",
    "load_failures",
    "settings",
]
