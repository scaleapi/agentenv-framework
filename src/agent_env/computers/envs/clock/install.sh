#!/bin/sh
# As root at compose: agent-env's libfaketime and its TLS shim, built from ./recipe and the downloaded tarball, preloaded
# into every process the computer starts, the library first, with the clock follower and its root loop beside them. The follower's user, agentenv-clock,
# owns the clock page, which core creates at arm time; clock.conf says so, so the library trusts no other page.
set -eu
sh recipe/build.sh /opt/agentenv/clock libfaketime.tar.gz
useradd --system --no-create-home --shell /usr/sbin/nologin agentenv-clock
install -m 0644 recipe/clockpage.py /opt/agentenv/clock/clockpage.py
install -m 0755 recipe/clock-follower /opt/agentenv/clock/clock-follower
install -m 0700 recipe/follow /opt/agentenv/clock/follow
install -d -m 0755 /etc/agentenv
sed "s/^AGENTENV_CLOCK_PAGE_UID=.*/AGENTENV_CLOCK_PAGE_UID=$(id -u agentenv-clock)/" recipe/clock.conf \
    > /etc/agentenv/clock.conf
chmod 0644 /etc/agentenv/clock.conf
printf '%s\n' /opt/agentenv/clock/libfaketimeMT.so.1 /opt/agentenv/clock/libtls_realtime.so > /etc/ld.so.preload
