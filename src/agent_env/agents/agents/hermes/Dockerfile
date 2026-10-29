FROM public.ecr.aws/docker/library/python:3.11-slim@sha256:c20888b6acdd1e63e1c433a185bf3ad162c0288fe484616ce062e0d28add2900
# Hermes Agent at a pinned release, installed by its own installer, which the image checks against its checksum. The
# installer brings Hermes' runtime (its Python, Node and ripgrep among them) under its home, /data/.hermes; git and xz
# are what it needs to fetch them. poppler-utils renders a PDF attachment to the page images Hermes looks at.
ARG HERMES_COMMIT=bf0203a794a07aac3ff764972ab9087baa4e4f4e
ARG HERMES_INSTALLER_SHA256=99cb3691d55019c6973a7acb775a915648666cc24e483ca827a1cdbb3531a674
RUN set -eu; \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git poppler-utils xz-utils \
      && rm -rf /var/lib/apt/lists/*; \
    useradd --create-home --uid 10001 agent && mkdir -p /app /data; \
    curl -fsSL --retry 5 --retry-all-errors \
      "https://raw.githubusercontent.com/NousResearch/hermes-agent/${HERMES_COMMIT}/scripts/install.sh" -o /tmp/install.sh; \
    echo "$HERMES_INSTALLER_SHA256  /tmp/install.sh" | sha256sum -c -; \
    HOME=/data bash /tmp/install.sh --commit "$HERMES_COMMIT" --dir /opt/hermes --hermes-home /data/.hermes \
      --non-interactive --skip-browser --skip-computer-use; \
    # What a turn never reads: the checkout's history, its UI builds' sources and dependencies, tests and docs, and
    # the installer's download cache.
    rm -rf /tmp/install.sh /opt/hermes/.git /opt/hermes/node_modules /opt/hermes/website /opt/hermes/tests \
      /data/.hermes/cache /data/.local /data/.bashrc /data/.profile; \
    ln -s /opt/hermes/.hermes/bin/hermes /usr/local/bin/hermes; \
    chown -R agent:agent /app /data
WORKDIR /opt/agent
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY a2a_server.py hermes_bridge.py hermes_kit.py entrypoint.sh ./
COPY agentenv_plugin ./agentenv_plugin
RUN chmod 0755 entrypoint.sh
# Each context's Hermes profile is under /data/.hermes, Hermes' home, whatever the uid: Modal runs the image as root.
ENV A2A_PORT=8000 PYTHONPATH=/opt/agent HOME=/data HERMES_ROOT=/data/.hermes
USER agent
WORKDIR /app
EXPOSE 8000
# CMD rather than ENTRYPOINT: Modal passes a sandbox's env to the image's CMD only.
CMD ["/opt/agent/entrypoint.sh"]
