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
# Every clocked agent's egress rule, which the bridge imports, and the clock page's format it reads (shared_sources.toml).
COPY agent_clock/agent_egress.py clock/clockpage.py ./
COPY agentenv_plugin ./agentenv_plugin
RUN chmod 0755 entrypoint.sh
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
# Each context's Hermes profile is under /data/.hermes, Hermes' home, whatever the uid.
ENV A2A_PORT=8000 PYTHONPATH=/opt/agent HOME=/data HERMES_ROOT=/data/.hermes
WORKDIR /app
EXPOSE 8000
# Starts as root, which start-agent needs for the clock; the A2A server and the CLI run as the agent's user. CMD rather
# than ENTRYPOINT: Modal passes a sandbox's env to the image's CMD only.
CMD ["/usr/local/bin/start-agent"]
