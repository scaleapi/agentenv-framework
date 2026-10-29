FROM public.ecr.aws/docker/library/python:3.11-slim@sha256:c20888b6acdd1e63e1c433a185bf3ad162c0288fe484616ce062e0d28add2900
# Node to install the OpenCode CLI, a native binary, and for the model's commands, checked against its published
# checksum; install.sh brings the same. poppler-utils renders a PDF attachment to the page images the model reads.
ARG NODE_VERSION=22.23.3
ARG NODE_SHA256_X64=1084aa36196bba4c3a5e69a1ee388a6e4ff729dad09445fbcd434b28fe3c24af
ARG NODE_SHA256_ARM64=5ced2d48d1d7198739b7f86804de0171aefb6823b684b12341d3321afc3cb0b2
RUN set -eu; \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl poppler-utils \
      && rm -rf /var/lib/apt/lists/*; \
    case "$(uname -m)" in x86_64) cpu=x64 sha256=$NODE_SHA256_X64 ;; aarch64) cpu=arm64 sha256=$NODE_SHA256_ARM64 ;; \
      *) echo "Node has no build for $(uname -m)" >&2; exit 1 ;; esac; \
    curl -fsSL --retry 3 "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${cpu}.tar.gz" \
      -o /tmp/node.tar.gz; \
    echo "$sha256  /tmp/node.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/node.tar.gz -C /usr/local --strip-components=1 --no-same-owner; \
    rm /tmp/node.tar.gz
WORKDIR /opt/agent
# The CLI as package-lock.json pins it: opencode-ai's own postinstall links its build for this CPU and libc into
# place, and the other builds the lockfile brings are removed.
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --omit=dev --no-audit --no-fund --loglevel=error && npm cache clean --force \
    && node node_modules/opencode-ai/postinstall.mjs \
    && find node_modules -mindepth 1 -maxdepth 1 -name 'opencode-*' ! -name opencode-ai -exec rm -rf {} + \
    && ln -s /opt/agent/node_modules/.bin/opencode /usr/local/bin/opencode
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# install.sh and run.sh are install/v1's: in the image, so they stay in its build context.
COPY a2a_server.py opencode_bridge.py opencode_kit.py entrypoint.sh install.sh run.sh ./
# The CLI's cwd, /app, and its state under /data, fixed so neither follows the uid: Modal runs the image as root.
RUN useradd --create-home --uid 10001 agent && mkdir -p /app /data \
    && chown agent:agent /app /data && chmod 0755 entrypoint.sh
ENV A2A_PORT=8000 PYTHONPATH=/opt/agent HOME=/data XDG_CONFIG_HOME=/data/config XDG_CACHE_HOME=/data/cache \
    XDG_DATA_HOME=/data/share XDG_STATE_HOME=/data/state
USER agent
WORKDIR /app
EXPOSE 8000
# CMD rather than ENTRYPOINT: Modal passes a sandbox's env to the image's CMD only.
CMD ["/opt/agent/entrypoint.sh"]
