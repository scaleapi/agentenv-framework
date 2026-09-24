"""Base `Env` contract.

`deploy` is part of the custom-env contract: the base raises `NotImplementedError`
(not `@abstractmethod`, which would block deserializing a read-only env) so the
`deploy_env` step fails clearly on an env that never implements it. Every built-in
env overrides `deploy`; this pins the base behavior a custom env inherits.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_env.env.env import Env


class _DeploylessEnv(Env):
    type = "deployless_env_test"

    @classmethod
    def from_dict(cls, data: dict) -> "_DeploylessEnv":
        return cls(id=data["id"], version=data.get("version"), metadata=data.get("metadata"))


def test_base_deploy_raises_not_implemented():
    env = _DeploylessEnv(id="e1", version=1)
    with pytest.raises(NotImplementedError, match="must implement deploy"):
        asyncio.run(env.deploy())
