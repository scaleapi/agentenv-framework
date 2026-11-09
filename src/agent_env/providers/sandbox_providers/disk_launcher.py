"""The QEMU launcher's interface, as its launch script reads it, for any provider that boots a computer's disk form
with it: the environment that sizes its VM, forwards its ports and hands the computer its variables, and its reset."""

import base64
import math

#: How the launcher takes the computer's environment, each variable under this prefix, base64'd (see its launch script).
ENV_PREFIX = "VM_ENV_"
#: The launch script's path in the launcher image, its ENTRYPOINT, which boots the computer; a provider that clears the
#: ENTRYPOINT runs it as the command.
LAUNCH = "/usr/local/bin/launch"
#: Where the launcher keeps what a reset and its boot loop share: a launcher older than reset in place has none.
CONTROL = "/var/lib/agentenv-vm/control"
#: The status ``RESET`` ends with in a sandbox whose launcher is older than reset in place.
UNRESETTABLE = 75
#: The command that boots the computer afresh, with the environment ``reset_input`` gives on its stdin: the launcher's
#: ``launch reset``, which a launcher older than it would take for a second boot, so it isn't run there.
RESET = ("sh", "-c", f"[ -d {CONTROL} ] || exit {UNRESETTABLE}; exec {LAUNCH} reset")


def forward(port: int) -> tuple[str, ...]:
    """The command that serves ``port`` of this machine's host to the running computer, as ``VM_HOST_PORTS`` does: the
    launcher's ``launch forward``, which a launcher older than it would take for a second boot, so it isn't run there,
    whose computer has no use for it."""
    return ("sh", "-c", f"grep -q '^forward()' {LAUNCH} || exit 0; exec {LAUNCH} forward {port}")


def launcher_variables(ports: list[int], env: dict[str, str], cpu: float, memory: int) -> dict[str, str]:
    """The environment that has the launcher size its VM, forward ``ports``, refuse to boot without KVM and hand the
    computer ``env``, each value base64'd, so one of several lines reaches the computer whole."""
    return {"VM_PORTS": " ".join(map(str, ports)), "VM_CPUS": str(max(1, math.ceil(cpu))), "VM_MEMORY_MB": str(memory),
            "VM_REQUIRE_KVM": "1", **_computer_variables(env)}


def reset_input(env: dict[str, str]) -> str:
    """What ``RESET`` reads on its stdin: the computer's new environment ``env``, a variable a line."""
    return "".join(f"{name}={value}\n" for name, value in _computer_variables(env).items())


def unresettable(computer: str) -> str:
    """Why ``computer``, booted by a launcher older than reset in place, isn't reset."""
    return f"{computer} was booted by an older agent-env, whose launcher can't reset it in place; redeploy it"


def _computer_variables(env: dict[str, str]) -> dict[str, str]:
    return {f"{ENV_PREFIX}{name}": base64.b64encode(value.encode()).decode() for name, value in env.items()}
