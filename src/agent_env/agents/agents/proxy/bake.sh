#!/bin/sh
# Bakes the proxy into an image as it's built, so an install into one of its containers only starts it: a private
# Python and a venv holding requirements.txt in /opt/agentenv-proxy, compiled to bytecode so each container doesn't
# compile it on first import, and a marker naming the requirements it holds, which install.sh checks. Plain sh, for
# images with no bash. A task image's Dockerfile ends with
#
#   COPY --from=ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 /uv /opt/agentenv-proxy/bin/uv
#   COPY agentenv-proxy /opt/agentenv-proxy/src
#   RUN sh /opt/agentenv-proxy/src/bake.sh
#
# where agentenv-proxy is this folder, copied into the image's build context.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
. "$here/versions.sh"

export UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_INSTALL_DIR="$BAKED/python" UV_CACHE_DIR="$BAKED/cache"
"$BAKED/bin/uv" venv --quiet --python "$PYTHON_VERSION" "$BAKED/venv"
"$BAKED/bin/uv" pip install --quiet --compile-bytecode --python "$BAKED/venv/bin/python" -r "$here/requirements.txt"
rm -rf "$BAKED/cache"
"$BAKED/venv/bin/python" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
  "$here/requirements.txt" > "$BAKED/baked"
