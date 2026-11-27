#!/bin/bash
# Install this agent from the folder this script is in, and start it on PORT, without an image of its own:
#
#   install.sh container NAME PORT     into the running container NAME, as root, with /app as the agent's workspace
#   install.sh host NAME PORT WORKDIR  onto this Linux host, as agent NAME, with WORKDIR as the agent's workspace
#
# It brings uv, checked against its published checksum, and the agent's requirements, the Antigravity SDK with its
# harness among them, in a private Python, so the target's own Python and packages are never touched. A target with no
# bash, which the model's commands run in, gets it from its own package manager. The model endpoint's LITELLM_API_KEY,
# LITELLM_BASE_URL and LITELLM_EXTRA_HEADERS, and AGENT_TRUST_ENV, the KEY='VALUE' lines of the CA variables the agent
# needs, come in the environment and are kept in a 0600 file the agent reads.
set -euo pipefail
umask 077

mode=$1 target=$2 port=$3
here=$(cd "$(dirname "$0")" && pwd)

UV_VERSION=0.13.1
UV_SHA256_X86_64=6c86040a8f045ab9adc9e3ec4cdafa03fd00e9a988ed2bf28971c7fc2d0debd0
UV_SHA256_AARCH64=345a8308e4d1835cdfaadf692836fd3137a6d45864773f05ad7bf59442b1ba16
PYTHON_VERSION=3.12
START_TIMEOUT_S=120
STOP_TIMEOUT_S=10
AGENT_FILES="a2a_server.py antigravity_bridge.py antigravity_date.py antigravity_kit.py requirements.txt run.sh agent_clock/agent_egress.py clock/clockpage.py model_relay/model_relay.py"

# put SRC DEST replaces DEST by a rename, so a binary an earlier install's agent still runs can be replaced.
case $mode in
  container)
    root=/opt/agentenv-agent-$port  # one per agent: two in a container listen on two ports
    workspace=/app
    on() { docker exec -u 0 "$target" sh -c "$1"; }
    put() { docker cp "$1" "$target:$2.new" > /dev/null && on "mv -f $(quoted "$2.new") $(quoted "$2")"; }
    ;;
  host)
    [ "$(uname -s)" = Linux ] || { echo "a host install needs a Linux host, not $(uname -s)" >&2; exit 1; }
    # Outside the work dir install_agent empties before each attempt, as a container's is, so a retry stops the agent an
    # earlier attempt started. It ends in the sandbox's id, which names what a local sandbox's teardown removes from
    # /tmp; a VM's host goes with the VM.
    root=/tmp/agentenv-agent-$target-$port${AGENTENV_SANDBOX:+-$AGENTENV_SANDBOX}
    workspace=$4
    on() { sh -c "$1"; }
    put() { cp "$1" "$2.new" && mv -f "$2.new" "$2"; }
    ;;
  *) echo "usage: install.sh container NAME PORT | install.sh host NAME PORT WORKDIR" >&2; exit 2 ;;
esac

if [ "$mode" = host ]; then
  # /tmp is shared, and the folder will hold the model key: it is this user's alone, and one another made is refused.
  if ! mkdir -m 0700 "$root" 2> /dev/null; then
    [ -d "$root" ] && [ ! -L "$root" ] && [ -O "$root" ] \
      || { echo "$root is already there and isn't a folder of this user's own" >&2; exit 1; }
    chmod 0700 "$root"
  fi
fi

case $(on 'uname -m') in
  x86_64|amd64) uv_cpu=x86_64 uv_sha256=$UV_SHA256_X86_64 ;;
  aarch64|arm64) uv_cpu=aarch64 uv_sha256=$UV_SHA256_AARCH64 ;;
  *) echo "the Antigravity SDK has no build for $(on 'uname -m')" >&2; exit 1 ;;
esac

if ! on 'command -v bash >/dev/null'; then
  on 'if command -v apt-get >/dev/null; then apt-get update -qq && apt-get install -y -qq --no-install-recommends bash
    elif command -v apk >/dev/null; then apk add -q bash
    elif command -v dnf >/dev/null; then dnf install -y -q bash
    elif command -v microdnf >/dev/null; then microdnf install -y bash
    else exit 1; fi' > /dev/null \
    || { echo "the model's commands run in bash: the target has none, and bash can't be added" >&2; exit 1; }
fi

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
# One more go when curl fails: --retry passes over some transient failures, such as a reset HTTP/2 stream (exit 92).
fetch() { curl -fsSL --retry 3 "$1" -o "$2" || { sleep 2; curl -fsSL --retry 3 "$1" -o "$2"; }; }
checked() {  # checked FILE SHA256: FILE is what was published, or the install stops
  [ "$(sha256 "$1")" = "$2" ] || { echo "$(basename "$1") doesn't match its published checksum" >&2; exit 1; }
}
quoted() { printf "'%s'" "$(printf %s "$1" | sed "s/'/'\\\\''/g")"; }
r=$(quoted "$root")  # $root, as the commands run in the target read it

# Beside this script, in the work dir the sandbox's teardown removes, so a killed install leaves nothing elsewhere.
stage=$(mktemp -d "$here/../stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT

uv_archive=uv-$uv_cpu-unknown-linux-musl.tar.gz
fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$uv_archive" "$stage/$uv_archive"
checked "$stage/$uv_archive" "$uv_sha256"
tar -xzf "$stage/$uv_archive" -C "$stage"
chmod 0755 "$stage/uv-$uv_cpu-unknown-linux-musl/uv"

# The model relay, in the agent's Python, reads the system's CA certificates, and a bare image has none: such a
# target's agent trusts certifi's, from that Python, unless the trust lines name a bundle. The SDK's harness, a Go
# binary, reads SSL_CERT_FILE too.
certifi=$root/venv/lib/python$PYTHON_VERSION/site-packages/certifi/cacert.pem
needs_cas=
on 'for f in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/ca-bundle.pem /etc/ssl/cert.pem
  do [ -s "$f" ] && exit 0; done; exit 1' || needs_cas=1
case ${AGENT_TRUST_ENV:-} in *SSL_CERT_FILE=*) needs_cas= ;; esac

# The model endpoint and CA, as the agent's environment, readable by its user only.
{
  for name in LITELLM_API_KEY LITELLM_BASE_URL LITELLM_EXTRA_HEADERS; do
    printf '%s=%s\n' "$name" "$(quoted "${!name:-}")"
  done
  [ -z "${AGENT_TRUST_ENV:-}" ] || printf '%s\n' "$AGENT_TRUST_ENV"
  [ -z "${AGENTENV_CONTROL_TOKEN:-}" ] || printf 'AGENTENV_CONTROL_TOKEN=%s\n' "$(quoted "$AGENTENV_CONTROL_TOKEN")"
  [ -z "$needs_cas" ] || printf 'SSL_CERT_FILE=%s\n' "$(quoted "$certifi")"
  # The SDK's conversations and skills, as the image keeps them under its home.
  printf 'A2A_PORT=%s\nANTIGRAVITY_WORKSPACE_DIR=%s\nANTIGRAVITY_HOME=%s\nHOME=%s\n' \
    "$(quoted "$port")" "$(quoted "$workspace")" "$(quoted "$root/data/.antigravity")" "$(quoted "$root/data")"
} > "$stage/agent.env"

# An earlier install's agent, a retried step's say, stops first: its files are replaced and its port is freed.
on "if [ -f $r/agent.pid ]; then pid=\$(cat $r/agent.pid); kill \$pid 2> /dev/null; i=0; \
  while [ \$i -lt $STOP_TIMEOUT_S ] && kill -0 \$pid 2> /dev/null; do sleep 1; i=\$((i + 1)); done; \
  kill -9 \$pid 2> /dev/null; rm -f $r/agent.pid; fi; true"

# The agent starts afresh, with no conversations or skills of one it replaces.
on "rm -rf $r/data && mkdir -p $r/bin $r/app $r/data $(quoted "$workspace")"
put "$stage/uv-$uv_cpu-unknown-linux-musl/uv" "$root/bin/uv"
put "$stage/agent.env" "$root/agent.env"
for file in $AGENT_FILES; do
  put "$here/$file" "$root/app/${file##*/}"
done
on "chmod 0600 $r/agent.env && chmod 0755 $r/bin/uv $r/app/run.sh"

# A private Python, uv's own build of it and never the target's, with the agent's requirements; no cache is left
# behind.
on "cd $r && export UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_INSTALL_DIR=$r/python \
  UV_CACHE_DIR=$r/cache/uv && bin/uv venv --quiet --clear --python $PYTHON_VERSION venv \
  && bin/uv pip install --quiet --python venv/bin/python -r app/requirements.txt \
  && rm -rf $r/cache && [ -s $(quoted "$certifi") ]"

if [ "$mode" = container ]; then
  docker exec -d -u 0 "$target" sh -c "exec $r/app/run.sh $r > $r/agent.log 2>&1"
else
  cd "$workspace"
  nohup "$root/app/run.sh" "$root" > "$root/agent.log" 2>&1 < /dev/null &
fi

# Done once the agent answers on its port; an agent that exits first fails the install with the end of its log.
health="import urllib.request; urllib.request.urlopen('http://127.0.0.1:$port/health', timeout=2)"
for _ in $(seq "$START_TIMEOUT_S"); do
  on "$r/venv/bin/python -c \"$health\"" 2> /dev/null && exit 0
  on "[ ! -f $r/agent.pid ] || kill -0 \$(cat $r/agent.pid)" 2> /dev/null || break
  sleep 1
done
echo "the agent didn't start; the end of its log:" >&2
on "tail -n 40 $r/agent.log" >&2 || true
exit 1
