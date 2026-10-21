FROM public.ecr.aws/docker/library/postgres:18-alpine@sha256:77f585114c32fbca283dc835b0596f4e52b51b4c6662d7810b2f4084f60a1873

# ServiceDBEnv - PostgreSQL database for MCP services
#
# Configuration via environment variables:
# - POSTGRES_USER (default: agentenv)
# - POSTGRES_PASSWORD (default: agentenv)
# - POSTGRES_DB (default: agentenv)
#
# MCP servers connect via DATABASE_URL:
# postgresql://agentenv:agentenv@servicedb:5432/agentenv

ENV POSTGRES_USER=agentenv
ENV POSTGRES_PASSWORD=agentenv
ENV POSTGRES_DB=agentenv

HEALTHCHECK --interval=5s --timeout=5s --retries=5 \
    CMD pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB} || exit 1

EXPOSE 5432
