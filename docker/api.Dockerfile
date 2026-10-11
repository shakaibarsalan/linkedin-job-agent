# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# API image — FastAPI. No browser, no session, nothing that talks to LinkedIn.
#
# Multi-stage: wheels are compiled in `builder` and only the resulting
# site-packages are copied into the runtime layer, so no compiler toolchain
# ships to production.
# ---------------------------------------------------------------------------

# ── Stage 1: build wheels ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential is needed for any package without an aarch64 wheel.
# libpq-dev is not required — asyncpg speaks the wire protocol directly.
RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY backend/requirements.txt .

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install -r requirements.txt

# ── Stage 2: runtime ───────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    MIGRATIONS_DIR=/app/migrations

# curl is used by the container HEALTHCHECK below. tini reaps zombies and
# forwards signals, so `docker stop` actually reaches uvicorn.
RUN apt-get update \
 && apt-get install --no-install-recommends -y curl tini \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Non-root, with a pinned UID so volume ownership is predictable.
ARG APP_UID=10001
RUN groupadd --gid ${APP_UID} agent \
 && useradd --uid ${APP_UID} --gid ${APP_UID} --create-home --shell /usr/sbin/nologin agent

WORKDIR /app
COPY --chown=agent:agent backend/app /app/app
COPY --chown=agent:agent database/migrations /app/migrations

# Docker initialises an empty *named* volume from the image's directory,
# including ownership — so creating these as `agent` here is what makes the
# volume writable by a non-root process without any entrypoint chown dance.
RUN mkdir -p /data/logs /data/resumes /data/profile \
 && chown -R agent:agent /data

USER agent

EXPOSE 8000

# Liveness only: a Postgres blip must not make Docker kill a healthy API.
# /health/ready is the one that checks the database.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
# Runs uvicorn via app/serve.py rather than the CLI. uvicorn's --log-config
# flag expects a parseable file, so there is no CLI way to say "leave logging
# alone" — only the Python API accepts log_config=None. See app/serve.py.
CMD ["python", "-m", "app.serve"]
