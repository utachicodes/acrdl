PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup setup-ml test lint format demo clean

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

setup-ml: setup
	$(BIN)/pip install -e ".[dev,ml]"

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src tests

format:
	$(BIN)/ruff check --fix src tests

demo:
	$(BIN)/acrdl demo

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
