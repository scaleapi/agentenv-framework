"""The trajectory formats beside native a run prescribes: ``[trajectories] formats`` in config.toml, else none."""

import pytest

from agent_env.config.errors import ConfigError
from tst.util.config import config_with_document


def test_a_run_prescribes_no_format_but_native_unless_config_names_some():
    assert config_with_document({}).get_trajectory_formats() == ()
    assert config_with_document({"trajectories": {}}).get_trajectory_formats() == ()


def test_trajectories_formats_names_the_formats_a_run_prescribes():
    assert config_with_document({"trajectories": {"formats": ["atif/1"]}}).get_trajectory_formats() == ("atif/1",)
    assert config_with_document({"trajectories": {"formats": []}}).get_trajectory_formats() == ()


@pytest.mark.parametrize("formats", ["atif/1", [1], [""], {"atif/1": True}])
def test_formats_that_arent_a_list_of_names_are_refused(formats):
    with pytest.raises(ConfigError, match=r"\[trajectories\] formats must be a list of trajectory format names"):
        config_with_document({"trajectories": {"formats": formats}}).get_trajectory_formats()
