"""The metadata every `put` records carries the installed distribution's version; the lookup
names the distribution, not the import package, so a rename that misses it would drop the key
silently because the caller swallows the error.
"""

from importlib.metadata import version

from agent_env.cli.utils import detect_base_metadata


def test_base_metadata_records_the_installed_version():
    assert detect_base_metadata()["agent_env_version"] == version("agentenv-framework")
