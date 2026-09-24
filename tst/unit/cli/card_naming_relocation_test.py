"""The card-naming relocation: helpers live in agent_env.utils.card_naming; the CLI re-exports are retired."""

import subprocess
import sys

from agent_env.cli import utils as cli_utils


def test_old_cli_path_no_longer_re_exports_the_helpers():
    # The compatibility re-exports are retired. The hub backend imports
    # the card-name family from agent_env.utils.card_naming and its pin is well
    # past the relocation, so nothing reads it from the CLI module any more.
    assert not hasattr(cli_utils, "card_name_from_github")
    assert not hasattr(cli_utils, "card_name_from_source")


def test_importing_card_naming_pulls_neither_cli_nor_providers():
    code = (
        "import sys; import agent_env.utils.card_naming; "
        "assert 'agent_env.cli' not in sys.modules, 'cli imported'; "
        "assert 'agent_env.providers' not in sys.modules, 'providers imported'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
