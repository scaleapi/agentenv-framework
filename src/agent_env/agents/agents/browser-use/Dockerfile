# docker build -t browser-use-a2a src/agent_env/agents/agents/browser-use
FROM mirror.gcr.io/library/node:22-bookworm-slim@sha256:4b150b48b4116e2a22f2c39cc27025fbf4f153e2078fd3b14d159fdb5c7279da AS node
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --omit=dev --no-audit --no-fund --loglevel=error

FROM mirror.gcr.io/library/python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337
COPY --from=node /usr/local/bin/node /usr/local/bin/node
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home agent
COPY --from=node /app/node_modules node_modules
COPY a2a_server.py agent.mjs package.json ./
USER agent
EXPOSE 8000
CMD ["python3", "a2a_server.py"]
