"""Every task of a bundle as an OpenEnv env (https://github.com/huggingface/OpenEnv): ``build_app`` returns the server
any OpenEnv client resets, steps and scores. The only part of agent-env that imports ``openenv``."""

from .app import build_app

__all__ = ["build_app"]
