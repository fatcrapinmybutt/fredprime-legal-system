.PHONY: help install install-dev install-all test test-coverage lint format type-check security-check clean docs build deploy

# Default target
help:
	@echo "FRED Supreme Litigation OS - Development Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install dev dependencies"
	@echo "  make install-all    - Install all dependencies"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test           - Run tests"
	@echo "  make test-coverage  - Run tests with coverage report"
	@echo "  make lint           - Run all linters"
	@echo "  make format         - Format code (Black, isort)"
	@echo "  make type-check     - Type checking with MyPy"
	@echo "  make security-check - Security checks (Bandit)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make check          - Run all checks (lint, type, security)"
	@echo "  make pre-commit     - Run pre-commit hooks"
	@echo ""
	@echo "Documentation & Building:"
	@echo "  make docs           - Generate documentation"
	@echo "  make build          - Build distribution packages"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make clean-cache    - Remove cache files"
	@echo "  make enforce-rulesets - Run custom ruleset enforcement"
	@echo ""

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[dev,docs]"

# Testing targets
test:
	pytest

test-coverage:
	pytest --cov=src --cov=modules --cov-report=html --cov-report=term

test-fast:
	pytest -m "not slow"

test-verbose:
	pytest -vv --tb=long

# Code quality targets
lint:
	@echo "Running Flake8..."
	flake8 src modules cli scripts tests --max-line-length=120
	@echo "Running isort check..."
	isort --check-only --diff src modules cli scripts tests
	@echo "Running Black check..."
	black --check --diff src modules cli scripts tests

format:
	@echo "Running isort..."
	isort src modules cli scripts tests
	@echo "Running Black..."
	black src modules cli scripts tests
	@echo "Code formatted successfully!"

type-check:
	mypy src modules cli scripts --ignore-missing-imports

security-check:
	bandit -r src modules cli scripts -ll

check: lint type-check security-check test
	@echo "✓ All checks passed!"

# Pre-commit
pre-commit:
	pre-commit run --all-files

pre-commit-install:
	pre-commit install

# Enforcement
enforce-rulesets:
	bash scripts/enforce_rulesets.sh

enforce-rulesets-strict:
	ENFORCE_STRICT=1 bash scripts/enforce_rulesets.sh

# Documentation
docs:
	@echo "Documentation generation would require sphinx setup"
	@echo "See docs/ directory for markdown docs"
	ls -la docs/

# Build & Distribution
build: clean
	python -m build

upload:
	python -m twine upload dist/*

# Cleanup targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete

clean-cache:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/

clean-all: clean clean-cache
	@echo "Cleaned all build and cache artifacts"

# Git hooks
git-hooks:
	pre-commit install

# Development environment
dev-setup: install-all git-hooks
	@echo "Development environment setup complete!"
	@echo "Run: make check  to verify setup"

# Quick validation
quick-check: format lint
	@echo "✓ Quick validation passed!"

# CI simulation
ci: clean install-all check test-coverage
	@echo "✓ CI pipeline simulation complete!"
