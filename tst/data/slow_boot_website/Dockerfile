FROM mirror.gcr.io/library/python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY server.py /app/server.py
ENV PORT=8000
EXPOSE 8000
RUN useradd --create-home --uid 10001 appuser
USER appuser
CMD ["python", "/app/server.py"]
