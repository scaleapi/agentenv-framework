# docker build -t browser-use-a2a src/agent_env/agents/agents/browser-use
FROM mirror.gcr.io/library/python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337
WORKDIR /app
COPY requirements.txt browser-use-requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m venv /opt/browser-use \
    && /opt/browser-use/bin/pip install --no-cache-dir -r browser-use-requirements.txt \
    && useradd --create-home agent
COPY a2a_server.py agent.py ./
USER agent
ENV ANONYMIZED_TELEMETRY=false BH_TELEMETRY=0
EXPOSE 8000
CMD ["python3", "a2a_server.py"]
