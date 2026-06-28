# ============================================================================
# DefaultRadar — developer entrypoints.
# Run `make help` for the list. Targets marked (Phase N) land in later phases.
# ============================================================================
.DEFAULT_GOAL := help
SHELL := /bin/bash

COMPOSE := docker compose
UV := uv run

# Load .env (if present) so host-side tools (make data/test) use the SAME ports
# as docker compose — otherwise MLFLOW_TRACKING_URI here could diverge from the
# port compose published.
ifneq (,$(wildcard .env))
include .env
export
endif

# Fallback only when .env is absent (matches config.py's default).
export MLFLOW_TRACKING_URI ?= http://localhost:5000

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Environment / dependencies --------------------------------------------
.PHONY: install
install: ## Create venv + install all deps from the lockfile (uv sync)
	uv sync --extra dev

# --- Docker stack -----------------------------------------------------------
.PHONY: up
up: ## Build + start the full stack (MLflow + Postgres + scorer + Prefect)
	$(COMPOSE) up -d --build
	@echo "MLflow  -> http://localhost:$${MLFLOW_PORT:-5000}"
	@echo "Scorer  -> http://localhost:$${SCORER_PORT:-8000}/docs"
	@echo "Prefect -> http://localhost:$${PREFECT_PORT:-4200}"

.PHONY: up-core
up-core: ## Start only Postgres + MLflow (lighter; the Phase-1 gate)
	$(COMPOSE) up -d --build postgres mlflow

.PHONY: ps
ps: ## Show stack status
	$(COMPOSE) ps

.PHONY: logs
logs: ## Tail logs from all services
	$(COMPOSE) logs -f

.PHONY: down
down: ## Stop the stack (keep volumes)
	$(COMPOSE) down

.PHONY: clean
clean: ## Stop the stack and remove volumes (DB, artifacts, prefect state)
	$(COMPOSE) down -v

# --- Data -------------------------------------------------------------------
.PHONY: data
data: ## Download + cache dataset, build Parquet, regen CI sample, print summary
	$(UV) python scripts/download_data.py

.PHONY: features
features: ## Build engineered, leakage-checked, time-split feature store (Parquet)
	$(UV) defaultradar features

.PHONY: summary
summary: ## Print the DuckDB base-rate/cohort summary (uses best available source)
	$(UV) defaultradar summary

.PHONY: sample
sample: ## (Re)generate the committed CI sample CSV from the full Parquet
	$(UV) defaultradar sample

# --- Quality ----------------------------------------------------------------
.PHONY: test
test: ## Run the test suite (integration tests auto-skip if services are down)
	$(UV) pytest

.PHONY: test-int
test-int: ## Run only integration tests (requires `make up`)
	$(UV) pytest -m integration

.PHONY: lint
lint: ## Lint with ruff
	$(UV) ruff check .

.PHONY: format
format: ## Auto-format with ruff
	$(UV) ruff format .
	$(UV) ruff check --fix .

# --- Lifecycle (later phases) ----------------------------------------------
.PHONY: train
train: ## Train + calibrate + log params/metrics/artifacts to MLflow + register a version
	$(UV) defaultradar train

.PHONY: eval
eval: ## Evaluate the registered model on test; non-zero exit if the gate is missed
	$(UV) defaultradar eval

.PHONY: promote
promote: ## Metric-gated Staging->Production promotion of the latest model version
	$(UV) defaultradar promote

.PHONY: serve
serve: ## Run the FastAPI scorer locally (dev; reloads on change)
	$(UV) uvicorn defaultradar.serving.app:app --host 0.0.0.0 --port $${SCORER_PORT:-8000} --reload

.PHONY: monitor
monitor: ## Run drift monitoring with injected distribution shift (PSI + Evidently)
	$(UV) defaultradar monitor

.PHONY: monitor-baseline
monitor-baseline: ## Run drift monitoring WITHOUT injected drift (baseline)
	$(UV) defaultradar monitor --no-inject

.PHONY: retrain
retrain: ## Retrain on the expanded window -> register -> gate -> promote
	$(UV) defaultradar retrain

.PHONY: demo
demo: ## End-to-end lifecycle: train -> promote -> drift -> retrain -> promote
	$(UV) defaultradar demo
