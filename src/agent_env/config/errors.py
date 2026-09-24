"""Errors for the ``.agentenv/config.toml`` configuration surface."""


class ConfigError(ValueError):
    """Raised when the ``.agentenv/config.toml`` (or a backend/section selection) is invalid."""

    pass


class EmptySecretBundleError(ConfigError, KeyError):
    """Empty secret bundle — also a KeyError so key-tolerant callers keep degrading."""

    __str__ = Exception.__str__  # KeyError.__str__ would repr-quote the message
