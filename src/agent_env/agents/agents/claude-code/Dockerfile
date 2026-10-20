# Node for the scripts the model runs, without npm or npx; Claude Code itself is a native build.
FROM public.ecr.aws/docker/library/node:22-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392 AS node

FROM public.ecr.aws/docker/library/python:3.11-slim@sha256:c20888b6acdd1e63e1c433a185bf3ad162c0288fe484616ce062e0d28add2900
# poppler-utils: the CLI's Read tool runs pdftoppm for a PDF's pages.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl poppler-utils \
    && rm -rf /var/lib/apt/lists/*
# Claude Code's native build for this machine's platform, checked against its release manifest.
ARG CLAUDE_CODE_VERSION=2.1.280
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
COPY a2a_server.py claude_bridge.py kit.py entrypoint.sh install.sh run.sh ./
# The CLI's cwd, /app, and its state, fixed so neither follows the uid: Modal runs the image as root.
RUN useradd --create-home --uid 10001 agent && mkdir -p /app /home/agent/.claude \
    && chown agent:agent /app /home/agent/.claude && chmod 0755 entrypoint.sh
ENV A2A_PORT=8000 PYTHONPATH=/opt/agent CLAUDE_CONFIG_DIR=/home/agent/.claude \
    DISABLE_AUTOUPDATER=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
USER agent
WORKDIR /app
EXPOSE 8000
# CMD rather than ENTRYPOINT: Modal passes a sandbox's env to the image's CMD only.
CMD ["/opt/agent/entrypoint.sh"]
