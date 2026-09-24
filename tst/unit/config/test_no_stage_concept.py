"""agent-env carries no stage concept: no stage kwarg, no stage attribute, no stage variable.

The stage is which config file ``AGENT_ENV_CONFIG`` resolves to, and only the installed
platform sdk knows how to choose between per-stage files. These are reintroduction guards —
the retired surfaces each looked harmless and each silently pointed writes at the wrong
database for the reader who trusted them.
"""

import pathlib

import pytest

from agent_env.config.runtime import Config, configure

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "agent_env"


def test_configure_rejects_a_stage_kwarg():
    with pytest.raises(TypeError, match="environment"):
        configure(environment="dev")


def test_config_carries_no_stage_attribute():
    cfg = Config()
    assert "environment" not in Config.__dataclass_fields__
    assert not hasattr(cfg, "environment")


def test_no_module_reads_a_stage_environment_variable():
    """A source scan, not a behavioural mock: the property under test is the absence of a
    call, and the failure it guards (a re-added stage read) is invisible until it repoints
    a store at run time."""
    offenders = sorted(
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if "AGENT_ENV_ENVIRONMENT" in p.read_text()
    )
    assert offenders == [], (
        f"stage reads are back in core: {offenders}. The stage is the config file "
        "AGENT_ENV_CONFIG selects; per-stage values belong in that file."
    )
