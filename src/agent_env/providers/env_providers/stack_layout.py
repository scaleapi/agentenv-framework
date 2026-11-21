"""Where a gateway stack lives in its sandbox: the default single stack, or one slot of several sharing a VM.

Imports nothing from agent_env but ``constants``, so modules in the ``agent_env.env`` import chain can use it while
``agent_env.providers`` is still importing.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from dataclasses import dataclass

from agent_env.providers.env_providers.constants import GATEWAY_APP_DIR

SLOT_HOST_PORT_STRIDE = 1000
MAX_PORT = 65535
_COMPOSE_PROJECT_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
_SHELL_SAFE_PATH = re.compile(r"/[A-Za-z0-9/_.-]*")
_DEFAULT_IMPLICIT_PROJECT = posixpath.basename(GATEWAY_APP_DIR)


def _check_shell_safe_path(field: str, path: str) -> None:
    """Paths are interpolated unquoted into shell scripts, so only absolute paths of a plain charset pass."""
    if not _SHELL_SAFE_PATH.fullmatch(path) or ".." in path.split("/"):
        raise ValueError(f"{field} {path!r} must be absolute, use only letters, digits, '/', '_', '.' or '-', and have no '..'")


@dataclass(frozen=True)
class StackLayout:
    """Where one gateway stack lives in its sandbox: its compose dir and project, its block of host ports and its scratch files.

    The default is the one stack a sandbox has always held. A slot is one of several stacks sharing a VM, so each gets its
    own dir, compose project, host ports and scratch names; container ports and service names are the same in every stack.
    """

    app_dir: str = GATEWAY_APP_DIR
    compose_project: str | None = None
    host_port_offset: int = 0
    scratch_prefix: str = "/tmp/"

    def __post_init__(self) -> None:
        _check_shell_safe_path("app dir", self.app_dir)
        _check_shell_safe_path("scratch prefix", self.scratch_prefix)
        if self.compose_project is not None:
            if not _COMPOSE_PROJECT_NAME.fullmatch(self.compose_project):
                raise ValueError(
                    f"compose project {self.compose_project!r} must be lowercase letters, digits, '-' or '_', starting with a letter or digit"
                )
            if self.compose_project == _DEFAULT_IMPLICIT_PROJECT:
                raise ValueError(
                    f"compose project {self.compose_project!r} is the default stack's implicit project (the basename of {GATEWAY_APP_DIR})"
                )
        if self.host_port_offset < 0 or self.host_port_offset % SLOT_HOST_PORT_STRIDE:
            raise ValueError(f"host port offset {self.host_port_offset} must be a non-negative multiple of {SLOT_HOST_PORT_STRIDE}")
        if self.compose_project is None and (
            self.app_dir != GATEWAY_APP_DIR or self.host_port_offset or self.scratch_prefix != "/tmp/"
        ):
            raise ValueError("a layout that differs from the default stack's must name its compose project")

    @property
    def is_slot(self) -> bool:
        """A slot names its compose project; only the default stack leaves it to compose."""
        return self.compose_project is not None

    @property
    def slot_index(self) -> int | None:
        """The index ``for_slot`` made this layout for; None for the default stack, or a slot layout built by hand."""
        index = (self.compose_project or "")[1:]
        if not (self.is_slot and self.compose_project.startswith("s") and index.isascii() and index.isdigit()):
            return None
        return int(index) if self == StackLayout.for_slot(int(index)) else None

    @classmethod
    def for_slot(cls, index: int) -> StackLayout:
        """Slot ``index`` of a shared VM: compose project ``s{index}`` names its dir and scratch files, its host ports are the index's block.

        The project is the hub's slot index, so the two cannot disagree. It is unique only within one docker daemon, which
        each Modal VM has to itself, so slots are for VMs, not local sandboxes sharing the host's daemon. Block 0 is the
        default stack's: a VM holds the default stack or slots, never both, but a slot starting one stride up keeps a
        mix-up from publishing on the default stack's ports. The gateway
        (18765) publishes on ``18765 + (index + 1) * 1000``, so 45 is the highest usable index; ``host_port`` raises past it.
        """
        if index < 0:
            raise ValueError(f"slot index must be >= 0, got {index}")
        project = f"s{index}"
        return cls(
            app_dir=f"{GATEWAY_APP_DIR}/{project}",
            compose_project=project,
            host_port_offset=(index + 1) * SLOT_HOST_PORT_STRIDE,
            scratch_prefix=f"/tmp/{project}-",
        )

    @property
    def compose_file(self) -> str:
        return f"{self.app_dir}/docker-compose.yml"

    @property
    def init_schemas_file(self) -> str:
        return f"{self.app_dir}/init-schemas.sql"

    @property
    def artifact_staging_dir(self) -> str:
        return f"{self.app_dir}/_artifact_staging"

    @property
    def compose_argv(self) -> tuple[str, ...]:
        """``docker compose`` bound to this stack's file and, for a slot, its project."""
        project = ("-p", self.compose_project) if self.is_slot else ()
        return ("docker", "compose", "-f", self.compose_file, *project)

    @property
    def compose_command(self) -> str:
        return " ".join(self.compose_argv)

    @property
    def compose_command_from_app_dir(self) -> str:
        """``docker compose`` for a script that has changed into ``app_dir``, where compose finds this stack's file."""
        project = f" -p {self.compose_project}" if self.is_slot else ""
        return f"docker compose{project}"

    def to_record(self) -> dict | None:
        """What a deploy record keeps of this layout, so a reload finds the same stack; None for the default stack."""
        if not self.is_slot:
            return None
        return {"app_dir": self.app_dir, "compose_project": self.compose_project,
                "host_port_offset": self.host_port_offset, "scratch_prefix": self.scratch_prefix}

    @classmethod
    def from_record(cls, data: object) -> StackLayout:
        """The layout a deploy record kept; the default stack for a record that kept none. Raises for one that isn't a
        valid slot layout, so a damaged record never sends a slot's commands to the default stack."""
        if data is None:
            return DEFAULT_STACK_LAYOUT
        fields = ("app_dir", "compose_project", "host_port_offset", "scratch_prefix")
        if not isinstance(data, dict) or set(data) != set(fields):
            raise ValueError(f"a recorded stack layout must have exactly the fields {fields}, got {data!r}")
        if not isinstance(data["host_port_offset"], int) or isinstance(data["host_port_offset"], bool):
            raise ValueError(f"a recorded stack layout's host_port_offset must be an int, got {data['host_port_offset']!r}")
        layout = cls(**data)
        if not layout.is_slot:
            raise ValueError(f"a recorded stack layout must be a slot's, got {data!r}")
        return layout

    @property
    def teardown_script(self) -> str:
        """A root shell script that removes this slot's stack and nothing else on the VM: every container, volume and
        network its compose project labels, then its dir and scratch files. Images stay, since every stack on the VM
        shares them. Idempotent, and it needs no compose file, so it also clears a slot whose deploy failed before
        writing one. Raises for the default stack, which has the VM to itself and goes with it, and for any layout
        not made by ``for_slot``: a hand-built project such as ``s1-x`` would have its files caught by ``s1``'s glob."""
        if not self.is_slot:
            raise ValueError("only a slot's stack is torn down on its own; the default stack ends with its VM")
        if self.slot_index is None:
            raise ValueError(f"only a layout made by StackLayout.for_slot is torn down, so its dir and scratch glob "
                             f"can't reach another stack's files; got {self}")
        label = shlex.quote(f"label=com.docker.compose.project={self.compose_project}")
        return "\n".join([
            "set -eu",
            f"containers=$(docker ps -aq --filter {label})",
            '[ -z "$containers" ] || docker rm -f -v $containers >/dev/null',
            f"volumes=$(docker volume ls -q --filter {label})",
            '[ -z "$volumes" ] || docker volume rm -f $volumes >/dev/null',
            f"networks=$(docker network ls -q --filter {label})",
            '[ -z "$networks" ] || docker network rm $networks >/dev/null',
            f"rm -rf {shlex.quote(self.app_dir)} {shlex.quote(self.scratch_prefix)}*",
        ])

    def scratch_path(self, name: str) -> str:
        return f"{self.scratch_prefix}{name}"

    def host_port(self, port: int) -> int:
        """The host port this stack publishes ``port`` on; raises if the shift leaves the port range."""
        shifted = port + self.host_port_offset
        if shifted > MAX_PORT:
            raise ValueError(f"port {port} shifted by {self.host_port_offset} is past {MAX_PORT}")
        return shifted


DEFAULT_STACK_LAYOUT = StackLayout()
