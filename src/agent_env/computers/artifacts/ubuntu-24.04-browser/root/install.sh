#!/bin/sh
# Installs the root start, as root at the image's build, the same in every base that starts as root (a unit test keeps
# the copies identical), once the Dockerfile has copied the clock's and the egress proxy's files into place: their users,
# root's folders and the clock's config folder, the session user's to neither read nor write, agentenv's trust files,
# and the clock library, preloaded into every process. $1 is the session user.
set -eu
session=$1
for file in /usr/local/lib/agentenv-clock/libfaketimeMT.so.1 /usr/local/lib/agentenv-clock/libtls_realtime.so \
        /usr/local/lib/agentenv-clock/clockpage.py /usr/local/lib/agentenv-clock/clock-follower \
        /etc/agentenv/clock.conf /usr/local/lib/agentenv-egress/egress /usr/local/lib/agentenv-root/init; do
    [ -f "$file" ] || { echo "the root start needs $file, which the base's Dockerfile copies" >&2; exit 2; }
done
useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin agentenv-clock
useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin agentenv-egress
mkdir -p /var/log/agentenv-root /run/agentenv-root /etc/agentenv/clock
chmod 0700 /var/log/agentenv-root /run/agentenv-root
chown root:agentenv-clock /etc/agentenv/clock && chmod 0750 /etc/agentenv/clock
touch /etc/agentenv/ca.pem
cp /etc/ssl/certs/ca-certificates.crt /etc/agentenv/ca-bundle.pem
cp /etc/ssl/certs/ca-certificates.crt /usr/local/lib/agentenv-egress/roots.pem
sed -i -e "s/^AGENTENV_SESSION_UID=.*/AGENTENV_SESSION_UID=$(id -u "$session")/" \
    -e "s/^AGENTENV_CLOCK_PAGE_UID=.*/AGENTENV_CLOCK_PAGE_UID=$(id -u agentenv-clock)/" /etc/agentenv/clock.conf
chmod 0755 /usr/local/lib/agentenv-clock/clock-follower /usr/local/lib/agentenv-root/init
printf '%s\n' /usr/local/lib/agentenv-clock/libfaketimeMT.so.1 /usr/local/lib/agentenv-clock/libtls_realtime.so \
    > /etc/ld.so.preload
