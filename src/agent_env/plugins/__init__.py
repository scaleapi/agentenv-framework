"""Entry-point plugins: register types from an installed package.

An installed distribution adds a class to a registry by declaring an entry point. The group
says what kind of thing it is, the entry-point name is the registry key, and the value is the
class::

    [project.entry-points."agent_env.envs"]
    browser = "agentenv_browser.env:BrowserEnv"

This package's top level is the public surface: the group names, ``PluginConflictError``,
``load_failures``, and ``inventory`` with the types it returns; its underscored modules are
internal. The full contract is in the README ("Register types from an installed package").
"""

from agent_env.plugins._inventory import Contribution, Distribution, Inventory, Status, inventory
from agent_env.plugins._registration import (
    ARTIFACTS,
    ENVS,
    EXPLORER_PLUGINS,
    SANDBOX_PROVIDERS,
    STATE_PROVIDERS,
    TASK_STEPS,
    PluginConflictError,
    load_failures,
)

__all__ = [
    "ARTIFACTS",
    "ENVS",
    "EXPLORER_PLUGINS",
    "SANDBOX_PROVIDERS",
    "STATE_PROVIDERS",
    "TASK_STEPS",
    "Contribution",
    "Distribution",
    "Inventory",
    "PluginConflictError",
    "Status",
    "inventory",
    "load_failures",
]
