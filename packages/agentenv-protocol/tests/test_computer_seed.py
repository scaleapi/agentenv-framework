"""The disk boot seed's version 1, as its sample pins it: what a launcher writes for an environment and what an init
reads back, every byte of a value kept, and where each version's user-data goes."""

from agentenv_protocol.computer import seed


def test_version_1_holds_the_sample_environment_byte_for_byte_and_reads_back_whole():
    assert seed.user_data(seed.SAMPLE) == seed.SAMPLE_USER_DATA
    assert seed.read_user_data(seed.SAMPLE_USER_DATA) == seed.SAMPLE


def test_each_version_has_its_own_file_and_version_1_keeps_nocloud_s_name():
    assert [seed.user_data_file(version) for version in (1, 2, 10)] == ["user-data", "user-data.v2", "user-data.v10"]
    assert seed.WRITTEN[0] == 1
