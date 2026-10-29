FROM public.ecr.aws/docker/library/python:3.11-slim@sha256:c20888b6acdd1e63e1c433a185bf3ad162c0288fe484616ce062e0d28add2900
# Node to install the Letta Code CLI, a native binary, and for the model's commands, checked against its published
# checksum; install.sh brings the same. poppler-utils renders a PDF attachment to the page images the model reads.
ARG NODE_VERSION=22.23.4
ARG NODE_SHA256_X64=b8492284c76a54baf58d61668dd0e41ee0fc98adabf9503d30f05d08f55b4406
ARG NODE_SHA256_ARM64=f7a1e486dbc8b286847815b477d4fb5e9fb963ed4489c3e1588853d685f589f9
RUN set -eu; \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl poppler-utils git \
      && rm -rf /var/lib/apt/lists/*; \
    case "$(uname -m)" in x86_64) cpu=x64 sha256=$NODE_SHA256_X64 ;; aarch64) cpu=arm64 sha256=$NODE_SHA256_ARM64 ;; \
      *) echo "Node has no build for $(uname -m)" >&2; exit 1 ;; esac; \
    curl -fsSL --retry 3 "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${cpu}.tar.gz" \
      -o /tmp/node.tar.gz; \
    echo "$sha256  /tmp/node.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/node.tar.gz -C /usr/local --strip-components=1 --no-same-owner; \
    rm /tmp/node.tar.gz
WORKDIR /opt/agent
# node-pty is interactive-only and has no Linux prebuild. Headless needs only these postinstalls.
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --omit=dev --no-audit --no-fund --loglevel=error \
    && npm rebuild @letta-ai/letta-code sharp protobufjs \
    && npm cache clean --force \
    && ln -s /opt/agent/node_modules/.bin/letta /usr/local/bin/letta
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# install.sh and run.sh are install/v1's: in the image, so they stay in its build context.
COPY mcp_tools.ts a2a_server.py letta_code_bridge.py letta_code_kit.py entrypoint.sh install.sh run.sh ./
# The CLI's cwd, /app, and its state under /data, fixed so neither follows the uid: Modal runs the image as root.
RUN useradd --create-home --uid 10001 agent && mkdir -p /app /data \
    && chown agent:agent /app /data && chmod 0755 entrypoint.sh \
    && find /opt/agent \( ! -perm -044 -o \( -perm /111 ! -perm -011 \) \) -exec chmod go+rX {} +
ENV A2A_PORT=8000 PYTHONPATH=/opt/agent HOME=/data XDG_CONFIG_HOME=/data/config XDG_CACHE_HOME=/data/cache \
    XDG_DATA_HOME=/data/share XDG_STATE_HOME=/data/state
USER agent
WORKDIR /app
EXPOSE 8000
# CMD rather than ENTRYPOINT: Modal passes a sandbox's env to the image's CMD only.
CMD ["/opt/agent/entrypoint.sh"]
