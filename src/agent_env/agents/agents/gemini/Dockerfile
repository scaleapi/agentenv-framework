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
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# install.sh and run.sh are install/v1's: in the image, so they stay in its build context.
COPY a2a_server.py gemini_bridge.py gemini_changelog.py gemini_config.py gemini_kit.py entrypoint.sh install.sh run.sh ./
# The CLI's cwd and file-tool boundary, /workspace, and its state, fixed so neither follows the uid: Modal runs the
# image as root. /app is kept, empty and writable, for tasks that name it.
RUN useradd --create-home --uid 10001 agent && mkdir -p /workspace/inputs /app /home/agent/.gemini \
    && chown agent:agent /workspace /workspace/inputs /app /home/agent/.gemini && chmod 0755 entrypoint.sh
ENV A2A_PORT=8000 PYTHONPATH=/opt/agent GEMINI_CLI_HOME=/home/agent GEMINI_CWD=/workspace
USER agent
WORKDIR /workspace
EXPOSE 8000
# CMD rather than ENTRYPOINT: Modal passes a sandbox's env to the image's CMD only.
CMD ["/opt/agent/entrypoint.sh"]
