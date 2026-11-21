# Node for the scripts the model runs, without npm or npx; Claude Code itself is a native build.
FROM public.ecr.aws/docker/library/node:22-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392 AS node

# The env's clock: the egress proxy, one static Go binary, and agent-env's libfaketime and its TLS shim, taken from
# agent-env's sources as clock/ and egress/ (shared_sources.toml) and built for this image's libc (Debian bookworm) in
# stages of their own, so no toolchain is left in the image.
FROM public.ecr.aws/docker/library/golang:1.26.11-bookworm@sha256:dc40a13b61304647ed7e5d581d246b86413dfc29d63c6d018310feb93cd98fc0 AS egress
COPY egress/ /egress/
RUN cd /egress && go vet ./... && go test -count=1 ./... \
    && CGO_ENABLED=0 GOFLAGS=-mod=readonly go build -trimpath -ldflags=-s -o /out/egress .

FROM public.ecr.aws/docker/library/python:3.11-slim@sha256:c20888b6acdd1e63e1c433a185bf3ad162c0288fe484616ce062e0d28add2900 AS clock
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential patch curl ca-certificates
COPY clock/ /clock/
RUN sh /clock/build.sh /out

FROM public.ecr.aws/docker/library/python:3.11-slim@sha256:c20888b6acdd1e63e1c433a185bf3ad162c0288fe484616ce062e0d28add2900
# poppler-utils: the CLI's Read tool runs pdftoppm for a PDF's pages.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl poppler-utils \
    && rm -rf /var/lib/apt/lists/*
# Claude Code's native build for this machine's platform, checked against its release manifest. a2a_server's _usage
# relies on this version's usage reports: `modelUsage` counts every call, subagents' too, and a resume restores the
# session's; each `result` event's `usage` counts the main agent's calls alone. Check both against a proxy's per-call
# usage, in a chat whose first turn uses a subagent, when bumping this.
ARG CLAUDE_CODE_VERSION=2.1.322
RUN set -eu; \
    case "$(uname -m)" in x86_64) platform=linux-x64 ;; aarch64) platform=linux-arm64 ;; \
      *) echo "Claude Code has no build for $(uname -m)" >&2; exit 1 ;; esac; \
    release="https://downloads.claude.ai/claude-code-releases/${CLAUDE_CODE_VERSION}"; \
    curl -fsSL --retry 3 "$release/manifest.json" -o /tmp/manifest.json; \
    checksum=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["platforms"][sys.argv[2]]["checksum"])' \
      /tmp/manifest.json "$platform"); \
    curl -fsSL --retry 3 "$release/$platform/claude" -o /usr/local/bin/claude; \
    echo "$checksum  /usr/local/bin/claude" | sha256sum -c -; \
    chmod 0755 /usr/local/bin/claude; \
    rm /tmp/manifest.json
COPY --from=node /usr/local/bin/node /usr/local/bin/node
WORKDIR /opt/agent
COPY requirements.txt model-requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r model-requirements.txt
# install.sh and run.sh are install/v1's: in the image, so they stay in its build context.
COPY a2a_server.py claude_bridge.py kit.py tar_changelog.py entrypoint.sh install.sh run.sh ./
# Every clocked agent's egress rule, which the bridge imports, and the clock page's format it reads (shared_sources.toml).
COPY agent_clock/agent_egress.py clock/clockpage.py ./
# The CLI's cwd, /app, and its state, fixed so neither follows the uid.
RUN useradd --create-home --uid 10001 agent && mkdir -p /app /home/agent/.claude \
    && chown agent:agent /app /home/agent/.claude && chmod 0755 entrypoint.sh
# The env's clock: the follower and the egress proxy, each as its own user, under root's supervisord (root.conf), and
# agent-env's libfaketime and its TLS shim preloaded into every process, faking only the agent's (clock.conf). The
# follower's config is the agent's user's to write (agentenv_protocol.a2a_agent.clock) and the follower's to read.
RUN pip install --no-cache-dir supervisor==4.2.5 \
    && useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin agentenv-clock \
    && useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin agentenv-egress \
    && mkdir -p /usr/local/lib/agentenv-clock /usr/local/lib/agentenv-egress /etc/agentenv \
    && install -d -o agent -g agentenv-clock -m 2750 /etc/agentenv/clock \
    && touch /etc/agentenv/ca.pem \
    && cp /etc/ssl/certs/ca-certificates.crt /etc/agentenv/ca-bundle.pem \
    && cp /etc/ssl/certs/ca-certificates.crt /usr/local/lib/agentenv-egress/roots.pem
COPY --from=clock /out/libfaketimeMT.so.1 /out/libtls_realtime.so /usr/local/lib/agentenv-clock/
COPY --from=egress /out/egress /usr/local/lib/agentenv-egress/egress
COPY clock/clockpage.py clock/clock-follower /usr/local/lib/agentenv-clock/
COPY clock/clock.conf /etc/agentenv/clock.conf
COPY root.conf /etc/agentenv/root.conf
COPY start-agent /usr/local/bin/start-agent
RUN sed -i -e "s/^AGENTENV_SESSION_UID=.*/AGENTENV_SESSION_UID=$(id -u agent)/" \
        -e "s/^AGENTENV_CLOCK_PAGE_UID=.*/AGENTENV_CLOCK_PAGE_UID=$(id -u agentenv-clock)/" /etc/agentenv/clock.conf \
    && chmod 0755 /usr/local/lib/agentenv-clock/clock-follower /usr/local/bin/start-agent \
    && printf '%s\n' /usr/local/lib/agentenv-clock/libfaketimeMT.so.1 /usr/local/lib/agentenv-clock/libtls_realtime.so \
        > /etc/ld.so.preload
ENV A2A_PORT=8000 PYTHONPATH=/opt/agent CLAUDE_CONFIG_DIR=/home/agent/.claude \
    DISABLE_AUTOUPDATER=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
WORKDIR /app
EXPOSE 8000
# Starts as root, which start-agent needs for the clock; the A2A server and the CLI run as the agent's user. CMD rather
# than ENTRYPOINT: Modal passes a sandbox's env to the image's CMD only.
CMD ["/usr/local/bin/start-agent"]
