#!/bin/bash
# Installs the proxy from the folder this script is in, and starts it on PORT, in the place of the agent it replaces:
#
#   install.sh host NAME PORT WORKDIR USER   onto this Linux host, as USER, in WORKDIR
#   install.sh container NAME PORT           into the running container NAME, as root, in its working directory
#
# On a host it runs in a venv of the host's own Python, which apt installs when the host has none. In a container it
# brings uv, checked against its published checksum and kept on the host, and a private Python, so the container's own
# are never touched; in one whose image bake.sh baked the proxy into, for these requirements, it only starts it.
set -euo pipefail

mode=$1 name=$2 port=$3
here=$(cd "$(dirname "$0")" && pwd)
. "$here/versions.sh"
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

case $mode in
  host)
    workdir=$4 user=$5
    root=/opt/agentenv-proxy-$name
    if ! python3 -c "import ensurepip, venv" 2>/dev/null; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq
      apt-get install -y -qq --no-install-recommends python3 python3-venv
    fi
    rm -rf "$root"
    mkdir -p "$root"
    cp "$here/a2a_server.py" "$here/requirements.txt" "$root/"
    python3 -m venv "$root/venv"
    "$root/venv/bin/pip" install -q -r "$root/requirements.txt"
    id -u "$user" >/dev/null 2>&1 || useradd -m "$user"
    mkdir -p "$workdir"
    chown -R "$user": "$workdir"
    nohup runuser -u "$user" -- env -C "$workdir" A2A_PORT="$port" "$root/venv/bin/python" "$root/a2a_server.py" \
      >"/tmp/agentenv-proxy-$name.log" 2>&1 </dev/null &
    ;;
  container)
    target=$name
    root=/opt/agentenv-proxy-$port  # one per port: two agents in a container listen on two
    on() { docker exec -u 0 "$target" sh -c "$1"; }
    # An earlier install's proxy, a retried step's say, stops first, so its port is free.
    on "if [ -f $root/proxy.pid ]; then kill \$(cat $root/proxy.pid) 2>/dev/null || true; fi; rm -rf $root; mkdir -p $root/bin"
    if [ "$(on "cat $BAKED/baked 2>/dev/null" || true)" = "$(sha256 "$here/requirements.txt")" ]; then
      python=$BAKED/venv/bin/python
    else
      case $(on 'uname -m') in
        x86_64|amd64) uv_cpu=x86_64 uv_sha256=$UV_SHA256_X86_64 ;;
        aarch64|arm64) uv_cpu=aarch64 uv_sha256=$UV_SHA256_AARCH64 ;;
        *) echo "uv has no build for $(on 'uname -m')" >&2; exit 1 ;;
      esac
      stage=$(mktemp -d "$here/../stage.XXXXXX")
      trap 'rm -rf "$stage"' EXIT
      uv_archive=uv-$uv_cpu-unknown-linux-musl.tar.gz
      cached=${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}/agentenv-proxy/$UV_VERSION-$uv_archive
      if [ ! -f "$cached" ] || [ "$(sha256 "$cached")" != "$uv_sha256" ]; then
        mkdir -p "$(dirname "$cached")"
        curl -fsSL --retry 3 "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$uv_archive" -o "$stage/$uv_archive"
        [ "$(sha256 "$stage/$uv_archive")" = "$uv_sha256" ] || { echo "$uv_archive doesn't match its published checksum" >&2; exit 1; }
        mv -f "$stage/$uv_archive" "$cached.$$" && mv -f "$cached.$$" "$cached"
      fi
      tar -xzf "$cached" -C "$stage"
      docker cp "$stage/uv-$uv_cpu-unknown-linux-musl/uv" "$target:$root/bin/uv" >/dev/null
      docker cp "$here/requirements.txt" "$target:$root/requirements.txt" >/dev/null
      on "cd $root && export UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_INSTALL_DIR=$root/python UV_CACHE_DIR=$root/cache \
        && bin/uv venv --quiet --python $PYTHON_VERSION venv && bin/uv pip install --quiet --python venv/bin/python \
        -r requirements.txt && rm -rf cache"
      python=$root/venv/bin/python
    fi
    docker cp "$here/a2a_server.py" "$target:$root/a2a_server.py" >/dev/null
    docker exec -d -u 0 -e A2A_PORT="$port" "$target" \
      sh -c "echo \$\$ > $root/proxy.pid; exec $python $root/a2a_server.py > $root/proxy.log 2>&1"
    ;;
  *) echo "usage: install.sh host NAME PORT WORKDIR USER | install.sh container NAME PORT" >&2; exit 2 ;;
esac
