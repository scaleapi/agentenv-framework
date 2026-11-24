"""Image loads go in scripts small enough for a provider's exec argument limit (Modal's VM sandbox: 65536 bytes)."""

from agent_env.providers.sandbox_providers.sandbox import _MAX_LOAD_SCRIPT_BYTES, _load_scripts


def _command(i: int, url_bytes: int = 2400) -> str:
    # The shape _load_docker_images writes: a curl of a signed URL, then docker load.
    return f"(curl -fsSL 'https://bucket.s3.amazonaws.com/img{i}.tar.gz?{'x' * url_bytes}' -o /tmp/i{i} && docker load)"


def test_a_few_images_stay_in_one_script():
    commands = [_command(i) for i in range(3)]
    assert _load_scripts(commands) == [" & ".join(commands) + " & wait"]


def test_many_images_split_under_the_limit_and_keep_every_command_once_in_order():
    commands = [_command(i) for i in range(36)]  # 14 MCP servers and 11 websites (two images each): about 86 KB
    scripts = _load_scripts(commands)
    assert len(scripts) > 1
    assert all(len(s.encode()) <= _MAX_LOAD_SCRIPT_BYTES for s in scripts)
    assert all(s.endswith(" & wait") for s in scripts)
    back = [c for s in scripts for c in s.removesuffix(" & wait").split(" & ")]
    assert back == commands


def test_a_command_longer_than_the_limit_gets_a_script_of_its_own():
    big = _command(0, url_bytes=_MAX_LOAD_SCRIPT_BYTES + 10)
    small = [_command(i) for i in range(1, 3)]
    scripts = _load_scripts([small[0], big, small[1]])
    assert scripts == [small[0] + " & wait", big + " & wait", small[1] + " & wait"]


def test_no_images_no_scripts():
    assert _load_scripts([]) == []
