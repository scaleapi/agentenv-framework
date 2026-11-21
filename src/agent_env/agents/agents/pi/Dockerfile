# docker build -t pi-a2a src/agent_env/agents/agents/pi
FROM mirror.gcr.io/library/ubuntu:24.04@sha256:73b999393d07f2bf65a40373778a44dc9a6d54ad7718d29247d8ff9413c4d6a8
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
