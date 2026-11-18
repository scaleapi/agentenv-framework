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
# Node for the Gemini CLI, checked against its published checksum; install.sh brings the same.
ARG NODE_VERSION=22.23.3
ARG NODE_SHA256_X64=1084aa36196bba4c3a5e69a1ee388a6e4ff729dad09445fbcd434b28fe3c24af
ARG NODE_SHA256_ARM64=5ced2d48d1d7198739b7f86804de0171aefb6823b684b12341d3321afc3cb0b2
RUN set -eu; \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*; \
    case "$(uname -m)" in x86_64) cpu=x64 sha256=$NODE_SHA256_X64 ;; aarch64) cpu=arm64 sha256=$NODE_SHA256_ARM64 ;; \
      *) echo "Node has no build for $(uname -m)" >&2; exit 1 ;; esac; \
    curl -fsSL --retry 3 "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${cpu}.tar.gz" \
      -o /tmp/node.tar.gz; \
    echo "$sha256  /tmp/node.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/node.tar.gz -C /usr/local --strip-components=1 --no-same-owner; \
    rm /tmp/node.tar.gz
WORKDIR /opt/agent
# The Gemini CLI, as package-lock.json pins it.
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --omit=dev --no-audit --no-fund --loglevel=error && npm cache clean --force \
    && ln -s /opt/agent/node_modules/.bin/gemini /usr/local/bin/gemini
COPY requirements.txt model-requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r model-requirements.txt
# install.sh and run.sh are install/v1's: in the image, so they stay in its build context.
COPY a2a_server.py gemini_bridge.py gemini_changelog.py gemini_config.py gemini_kit.py entrypoint.sh install.sh run.sh ./
# The CLI's cwd and file-tool boundary, /workspace, and its state, fixed so neither follows the uid. /app is kept,
# empty and writable, for tasks that name it.
RUN useradd --create-home --uid 10001 agent && mkdir -p /workspace/inputs /app /home/agent/.gemini \
    && chown agent:agent /workspace /workspace/inputs /app /home/agent/.gemini && chmod 0755 entrypoint.sh
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
ENV A2A_PORT=8000 PYTHONPATH=/opt/agent GEMINI_CLI_HOME=/home/agent GEMINI_CWD=/workspace
WORKDIR /workspace
EXPOSE 8000
# Starts as root, which start-agent needs for the clock; the A2A server and the CLI run as the agent's user. CMD rather
# than ENTRYPOINT: Modal passes a sandbox's env to the image's CMD only.
CMD ["/usr/local/bin/start-agent"]
