Base computers that come with agent-env, each a `computer` artifact a computer env composes a driver and apps onto.

`artifacts/ubuntu-24.04/` is Ubuntu 24.04 (`amd64` and `arm64`) with XFCE on Xvfb, D-Bus and AT-SPI. tini starts
supervisord, which runs every program as the session user, `user` (uid 1000); nothing in it runs as root. Its
container form's image is built from the folder's `Dockerfile`, and a release publishes it as
`ghcr.io/scaleapi/agentenv-ubuntu-24.04:src-<hex>`, tagged by its sources, so a run pulls it by digest instead of
building it. Name it from another bundle as `ubuntu-24.04`, the alias `aliases.toml` gives it, or by its id,
`@local/agentenv-framework/computers/ubuntu-24.04`.

Its settings reach the computer as environment variables, `AGENTENV_SETTING_<NAME>`, read as it starts:

| Setting | Default | What it does |
| --- | --- | --- |
| `resolution` | `1280x800` | The size of the Xvfb screen, as `<width>x<height>`, each 1 to 32767 pixels |
| `timezone` | `UTC` | The time zone every program runs in, a name under `/usr/share/zoneinfo` |

A value it can't use stops the computer at start, with the reason on stderr. Composing a driver or an app onto it adds
its services as supervisord programs in `/etc/supervisor/conf.d/`; they inherit the display (`DISPLAY=:99`), the
session bus and the time zone. A program that needs X runs as `start-desktop with-x <command>`, which waits for X, and
with `startsecs=0`, so X dying however often never leaves it `FATAL`, as the XFCE session does.

This bundle has no tasks of its own: other bundles pull its computers by alias, and `agent-env run computers` has
nothing to run.
