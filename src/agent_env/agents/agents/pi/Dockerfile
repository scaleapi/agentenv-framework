# docker build -t pi-a2a src/agent_env/agents/agents/pi
FROM mirror.gcr.io/library/ubuntu:24.04@sha256:2198e4c65076e9cb0b3d695219df667c0e3b0072da64ca3c2648aca17dcecaf5
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
