#!/bin/bash
# Resolve Ubuntu's archive to Azure's mirror of it for everything on this runner: its builds, its containers and the
# runner itself. A dnsmasq on Docker's bridge answers archive.ubuntu.com and security.ubuntu.com with the mirror's
# addresses and forwards every other name to the runner's own resolver; the runner's resolv.conf names it, and Docker
# copies that into each container and BuildKit into each RUN, since it isn't a loopback address.
set -euo pipefail
MIRROR=azure.archive.ubuntu.com
HOSTS=(archive.ubuntu.com security.ubuntu.com)
bridge=$(ip -4 addr show docker0 | awk '/inet /{print $2}' | cut -d/ -f1)
upstream=$(awk '/^nameserver/{print $2; exit}' /run/systemd/resolve/resolv.conf)
mirror=$(getent ahostsv4 "$MIRROR" | awk '{print $1; exit}')
[ -n "$bridge" ] && [ -n "$upstream" ] && [ -n "$mirror" ] || { echo "bridge=$bridge upstream=$upstream mirror=$mirror" >&2; exit 1; }
command -v dnsmasq >/dev/null || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq dnsmasq-base >/dev/null
args=(--conf-file=/dev/null --pid-file=/run/agentenv-archive-dns.pid --listen-address="$bridge" --bind-dynamic
      --no-resolv --server="$upstream" --filter-AAAA)
for host in "${HOSTS[@]}"; do args+=(--address="/$host/$mirror"); done
sudo dnsmasq "${args[@]}"
getent hosts "$(hostname)" >/dev/null || echo "127.0.1.1 $(hostname)" | sudo tee -a /etc/hosts >/dev/null
sudo rm -f /etc/resolv.conf
printf 'nameserver %s\noptions edns0\n' "$bridge" | sudo tee /etc/resolv.conf >/dev/null
for host in "${HOSTS[@]}"; do
  got=$(getent ahostsv4 "$host" | awk '{print $1; exit}')
  [ "$got" = "$mirror" ] || { echo "$host resolves to $got, not $MIRROR's $mirror" >&2; exit 1; }
done
getent ahostsv4 pypi.org >/dev/null || { echo "other names don't resolve" >&2; exit 1; }
echo "Ubuntu's archive resolves to $MIRROR ($mirror) through $bridge; other names through $upstream"
