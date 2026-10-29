#!/bin/bash
# Install this agent from the folder this script is in, and start it on PORT, without an image of its own:
#
#   install.sh container NAME PORT     into the running container NAME, as root, with /app as the CLI's workspace
#   install.sh host NAME PORT WORKDIR  onto this Linux host, as agent NAME, with WORKDIR as the CLI's workspace
#
# It brings Node and uv, each checked against its published checksum, the Letta Code CLI as package-lock.json pins it, and
# the agent's requirements in a private Python, so the target's own Node, Python and packages are never touched; a
# target with no Node or Python of its own gets these at the end of its PATH. Node's builds need glibc, and a target
# with no bash, which the model's commands run in, gets it from its own package manager. The model endpoint's
# LITELLM_API_KEY, LITELLM_BASE_URL and LITELLM_EXTRA_HEADERS, and AGENT_TRUST_ENV, the KEY='VALUE' lines of the CA
# variables the agent needs, come in the environment and are kept in a 0600 file the agent reads.
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
dockerfile_arg() { sed -n "s/^ARG $1=//p" "$here/Dockerfile"; }
NODE_VERSION=$(dockerfile_arg NODE_VERSION)
AGENT_FILES="mcp_tools.ts a2a_server.py letta_code_bridge.py letta_code_kit.py requirements.txt run.sh"

# put SRC DEST replaces DEST by a rename, so a binary an earlier install's agent still runs can be replaced; put_dir SRC
# DEST copies a folder to DEST, which isn't there.
case $mode in
  container)
    root=/opt/agentenv-agent-$port  # one per agent: two in a container listen on two ports
    workspace=/app
    on() { docker exec -u 0 "$target" sh -c "$1"; }
    put() { docker cp "$1" "$target:$2.new" > /dev/null && on "mv -f $(quoted "$2.new") $(quoted "$2")"; }
    put_dir() { docker cp "$1" "$target:$2" > /dev/null; }
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
    put_dir() { cp -R "$1" "$2"; }
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
  x86_64|amd64) cpu=x64 uv_cpu=x86_64 uv_sha256=$UV_SHA256_X86_64 node_sha256=$(dockerfile_arg NODE_SHA256_X64) ;;
  aarch64|arm64) cpu=arm64 uv_cpu=aarch64 uv_sha256=$UV_SHA256_AARCH64 node_sha256=$(dockerfile_arg NODE_SHA256_ARM64) ;;
  *) echo "Node has no build for $(on 'uname -m')" >&2; exit 1 ;;
esac
if on 'for f in /lib/ld-musl-*; do [ -e "$f" ] && exit 0; done; exit 1'; then
  echo "Node's builds need glibc, and the target's libc is musl" >&2
  exit 1
fi

if ! on 'command -v bash >/dev/null'; then
  on 'if command -v apt-get >/dev/null; then apt-get update -qq && apt-get install -y -qq --no-install-recommends bash
    elif command -v dnf >/dev/null; then dnf install -y -q bash
    elif command -v microdnf >/dev/null; then microdnf install -y bash
    else exit 1; fi' > /dev/null \
    || { echo "the model's commands run in bash: the target has none, and bash can't be added" >&2; exit 1; }
fi

# Letta Code keeps agent memory in git, including on freshly installed targets.
if ! on 'command -v git >/dev/null'; then
  on 'if command -v apt-get >/dev/null; then apt-get update -qq && apt-get install -y -qq --no-install-recommends git
    elif command -v dnf >/dev/null; then dnf install -y -q git
    elif command -v microdnf >/dev/null; then microdnf install -y git
    else exit 1; fi' > /dev/null \
    || { echo "Letta Code requires git, and git cannot be added" >&2; exit 1; }
fi

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
fetch() { curl -fsSL --retry 3 "$1" -o "$2"; }
checked() {  # checked FILE SHA256: FILE is what was published, or the install stops
  [ "$(sha256 "$1")" = "$2" ] || { echo "$(basename "$1") doesn't match its published checksum" >&2; exit 1; }
}
quoted() { printf "'%s'" "$(printf %s "$1" | sed "s/'/'\\\\''/g")"; }
r=$(quoted "$root")  # $root, as the commands run in the target read it

# Beside this script, in the work dir the sandbox's teardown removes, so a killed install leaves nothing elsewhere.
stage=$(mktemp -d "$here/../stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT

node_archive=node-v$NODE_VERSION-linux-$cpu.tar.gz
fetch "https://nodejs.org/dist/v$NODE_VERSION/$node_archive" "$stage/$node_archive"
checked "$stage/$node_archive" "$node_sha256"
mkdir "$stage/node"
tar -xzf "$stage/$node_archive" -C "$stage/node" --strip-components=1

uv_archive=uv-$uv_cpu-unknown-linux-musl.tar.gz
fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$uv_archive" "$stage/$uv_archive"
checked "$stage/$uv_archive" "$uv_sha256"
tar -xzf "$stage/$uv_archive" -C "$stage"
chmod 0755 "$stage/uv-$uv_cpu-unknown-linux-musl/uv"

# Use the private Node to execute the pinned CLI.
printf '#!/bin/sh\nexec %s %s "$@"\n' "$(quoted "$root/node/bin/node")" \
  "$(quoted "$root/npm/node_modules/@letta-ai/letta-code/letta.js")" > "$stage/letta"
chmod 0755 "$stage/letta"

# The model endpoint and CA, as the agent's environment, readable by its user only.
{
  for name in LITELLM_API_KEY LITELLM_BASE_URL LITELLM_EXTRA_HEADERS; do
    printf '%s=%s\n' "$name" "$(quoted "${!name:-}")"
  done
  [ -z "${AGENT_TRUST_ENV:-}" ] || printf '%s\n' "$AGENT_TRUST_ENV"
  # Letta Code's state, its sessions and skills, as the image keeps it under /data.
  printf 'A2A_PORT=%s\nLETTA_CODE_WORKSPACE_DIR=%s\nHOME=%s\n' \
    "$(quoted "$port")" "$(quoted "$workspace")" "$(quoted "$root/data")"
  for dir in CONFIG:config CACHE:cache DATA:share STATE:state; do
    printf 'XDG_%s_HOME=%s\n' "${dir%%:*}" "$(quoted "$root/data/${dir#*:}")"
  done
} > "$stage/agent.env"

# An earlier install's agent, a retried step's say, stops first: its files are replaced and its port is freed.
on "if [ -f $r/agent.pid ]; then pid=\$(cat $r/agent.pid); kill \$pid 2> /dev/null; i=0; \
  while [ \$i -lt $STOP_TIMEOUT_S ] && kill -0 \$pid 2> /dev/null; do sleep 1; i=\$((i + 1)); done; \
  kill -9 \$pid 2> /dev/null; rm -f $r/agent.pid; fi; true"

# The agent starts afresh, with no sessions or skills of one it replaces.
on "rm -rf $r/data $r/node $r/npm && mkdir -p $r/bin $r/app $r/data $r/npm $(quoted "$workspace")"
put_dir "$stage/node" "$root/node"
put "$stage/uv-$uv_cpu-unknown-linux-musl/uv" "$root/bin/uv"
put "$stage/letta" "$root/bin/letta"
put "$stage/agent.env" "$root/agent.env"
put "$here/package.json" "$root/npm/package.json"
put "$here/package-lock.json" "$root/npm/package-lock.json"
for file in $AGENT_FILES; do
  put "$here/$file" "$root/app/$file"
done
on "chmod 0600 $r/agent.env && chmod 0755 $r/bin/uv $r/bin/letta $r/app/run.sh"

# Install only headless dependencies' required scripts, then the private Python environment.
on "cd $r/npm && export PATH=$r/node/bin:\$PATH npm_config_cache=$r/cache/npm npm_config_update_notifier=false \
  && npm ci --ignore-scripts --omit=dev --no-audit --no-fund --loglevel=error > /dev/null \
  && npm rebuild @letta-ai/letta-code sharp protobufjs > /dev/null \
  && cd $r && export UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_INSTALL_DIR=$r/python \
  UV_CACHE_DIR=$r/cache/uv && bin/uv venv --quiet --clear --python $PYTHON_VERSION venv \
  && bin/uv pip install --quiet --python venv/bin/python -r app/requirements.txt \
  && rm -rf $r/cache"

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
