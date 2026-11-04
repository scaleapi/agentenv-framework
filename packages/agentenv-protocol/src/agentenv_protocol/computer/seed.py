"""The disk boot seed: how the launcher that boots a Linux computer's disk form hands it the deploy's environment, as a
NoCloud seed its init reads at boot. A disk composed by one agent-env version boots under another's launcher, so the
seed carries every version its launcher writes, and an init reads the newest it knows; a launcher that has stopped
writing every version an init reads says so, and that init stops the boot rather than start without its environment.

Version 1 is a FAT filesystem labelled ``CIDATA`` holding ``user-data``: ``HEADER``, then a line a variable,
``NAME=<its value, base64'd>``, ordered by line, so a value of any bytes stays one line. ``meta-data`` is NoCloud's own:
an ``instance-id`` and the guest's ``local-hostname``. A later version N goes in its own file, ``user-data.vN``, beside
the ones before it, and ``OLDEST`` names the oldest version the seed still carries."""

from __future__ import annotations

import base64

#: The label of the seed's FAT filesystem, which an init finds it by.
LABEL = "CIDATA"
#: The first line of every version's user-data.
HEADER = "#agentenv-environment"
#: The file holding the oldest version the seed carries, a decimal number. A seed without it carries version 1.
OLDEST = "agentenv-seed-oldest"
#: The versions a launcher of this protocol writes, oldest first.
WRITTEN = (1,)


def user_data_file(version: int) -> str:
    """The file ``version``'s user-data is in: ``user-data``, the name NoCloud gives it, for version 1, and
    ``user-data.v<version>`` after it."""
    return "user-data" if version == 1 else f"user-data.v{version}"


def user_data(environment: dict[str, str]) -> str:
    """Version 1's user-data for ``environment``."""
    lines = sorted(f"{name}={base64.b64encode(value.encode()).decode()}" for name, value in environment.items())
    return "".join(f"{line}\n" for line in [HEADER, *lines])


def read_user_data(text: str) -> dict[str, str]:
    """The environment version 1's ``text`` holds, as an init reads it: blank and ``#`` lines skipped."""
    return {name: base64.b64decode(value).decode() for line in text.splitlines() if line and not line.startswith("#")
            for name, _, value in [line.partition("=")]}


#: An environment and version 1's user-data for it, byte for byte, which a launcher's writer and an init's reader are
#: each tested against, wherever either is implemented.
SAMPLE = {"AGENTENV_CONTROL_TOKEN": "a value with spaces, = signs, 'quotes', $dollars and\na second line\n",
          "ENVIRONMENT_NAME": "desk", "ENVIRONMENT_NAME2": "", "NOTE": "first line\nFAKE=1"}
SAMPLE_USER_DATA = (
    "#agentenv-environment\n"
    "AGENTENV_CONTROL_TOKEN=YSB2YWx1ZSB3aXRoIHNwYWNlcywgPSBzaWducywgJ3F1b3RlcycsICRkb2xsYXJzIGFuZAphIHNlY29uZCBsaW5lCg==\n"
    "ENVIRONMENT_NAME2=\n"
    "ENVIRONMENT_NAME=ZGVzaw==\n"
    "NOTE=Zmlyc3QgbGluZQpGQUtFPTE=\n"
)
