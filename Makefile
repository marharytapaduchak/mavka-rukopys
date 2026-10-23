
# MAVKA — RUKOPYS
# Development setup and basic project checks.
# Model training and evaluation are performed in notebooks/.

PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup test lint clean

setup: ## Create virtual environment and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt -r requirements-dev.txt
	$(BIN)/pip install -e .
	$(BIN)/pre-commit install

test: ## Run project tests
	$(BIN)/python -m pytest tests/ -v

lint: ## Run pre-commit checks
	$(BIN)/pre-commit run --all-files

clean: ## Remove local Python cache files
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
