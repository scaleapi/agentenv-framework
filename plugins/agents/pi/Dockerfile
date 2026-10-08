# docker build -t pi-a2a plugins/agents/pi
FROM mirror.gcr.io/library/ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
WORKDIR /app
# Named sources: agent-env packs the build context install/v1 copies from the COPY sources.
COPY a2a_server.py changelog.py install.sh peers.py pi_agent.py requirements.txt start.sh ./
RUN sh install.sh \
    && useradd --create-home agent \
    && mkdir /workspace \
    && chown agent /app /workspace
ENV PATH="/opt/pi-a2a-venv/bin:/opt/pi/bin:/opt/node/bin:${PATH}"
USER agent
EXPOSE 8000
CMD ["python3", "a2a_server.py"]
