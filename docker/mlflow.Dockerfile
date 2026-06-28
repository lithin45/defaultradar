# MLflow tracking server with a Postgres backend store + served artifacts.
# Pinned to the same MLflow version as the project lockfile for reproducibility.
FROM python:3.12-slim

ARG MLFLOW_VERSION=2.22.5

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
        "mlflow==${MLFLOW_VERSION}" \
        "psycopg2-binary>=2.9"

# Physical artifact store (served back to clients via --serve-artifacts).
RUN mkdir -p /mlflow/artifacts
EXPOSE 5000

# The actual `mlflow server ...` command is supplied by docker-compose so the
# backend-store URI can be composed from environment variables.
