#!/bin/sh
# Builds agent-env's libfaketime from ./clock and the downloaded tarball, installs its settings and the page's writer,
# and preloads it into every process the computer starts.
set -eu
sh clock/build.sh /opt/agentenv/clock libfaketime.tar.gz
install -m 0644 clock/clockpage.py /opt/agentenv/clock/clockpage.py
install -D -m 0644 clock/clock.conf /etc/agentenv/clock.conf
echo /opt/agentenv/clock/libfaketimeMT.so.1 > /etc/ld.so.preload
