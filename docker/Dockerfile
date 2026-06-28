# syntax=docker/dockerfile:1
# ----------------------------------------------------------------------------
# DefaultRadar application image — shared by the FastAPI scorer, the Prefect
# server/flows, and (optionally) batch training. Built once, reused by both the
# `scorer` and `prefect` compose services.
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS base

# uv (pinned) for fast, reproducible, lockfile-driven installs.
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Runtime libs: libgomp1 for XGBoost's OpenMP, curl for compose healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dependencies only — a cache-friendly layer keyed on the lockfile. The uv
#    download cache is cleaned in the SAME step so it is never committed into the
#    image (keeps it ~2GB, not ~4GB, and shrinks the build footprint on small VMs).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev \
    && uv cache clean && rm -rf /root/.cache/uv

# 2) Project source — changes here only re-run the fast project install, not the
#    full dependency download.
COPY src ./src
COPY config ./config
COPY scripts ./scripts
RUN uv sync --frozen --no-dev && uv cache clean && rm -rf /root/.cache/uv

# Prefect stores its local DB / state here (mounted as a volume in compose).
ENV PREFECT_HOME=/app/.prefect

EXPOSE 8000 4200

# Default command is the scorer; compose overrides it for the Prefect service.
CMD ["uvicorn", "defaultradar.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
