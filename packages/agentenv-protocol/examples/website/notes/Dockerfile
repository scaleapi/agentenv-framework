FROM mirror.gcr.io/library/node:22-slim@sha256:4b150b48b4116e2a22f2c39cc27025fbf4f153e2078fd3b14d159fdb5c7279da AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM mirror.gcr.io/library/python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf
WORKDIR /app
RUN pip install --no-cache-dir "agentenv-framework-protocol>=0.1.466" "uvicorn>=0.30"
COPY site.py /app/site.py
COPY data /app/data
COPY --from=web /web/dist /app/web/dist
ENV PORT=8000
EXPOSE 8000
RUN useradd --create-home --uid 10001 appuser
USER appuser
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/.well-known/agent-env.json' % os.environ.get('PORT', '8000'), timeout=2)"
CMD ["python", "site.py"]
