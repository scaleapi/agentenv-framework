FROM public.ecr.aws/docker/library/node:22-slim@sha256:813a7480f28fdadac1f7f5c824bcdad435b5bc1322a5968bbbdef8d058f9dff4

# Chromium runtime dependencies for headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libwayland-client0 \
    fonts-liberation \
    fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG PLAYWRIGHT_MCP_VERSION
ENV PLAYWRIGHT_MCP_VERSION=${PLAYWRIGHT_MCP_VERSION}

# Pre-install the pinned version and Playwright's Chromium, which is built for linux/amd64 and linux/arm64 alike
# (Chrome has no Linux arm64 build).
RUN npm init -y && \
    npm install @playwright/mcp@${PLAYWRIGHT_MCP_VERSION} && \
    npx playwright install chromium

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
