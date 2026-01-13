#!/bin/bash
# Setup script for CI/CD environments

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "🚀 Setting up CI/CD environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
  echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
  echo -e "${RED}✗${NC} $1"
}

# Check Python version
check_python() {
  if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is required but not installed."
    exit 1
  fi

  PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
  log_info "Found Python $PYTHON_VERSION"
}

# Setup virtual environment
setup_venv() {
  if [ ! -d .venv ]; then
    log_info "Creating virtual environment..."
    python3 -m venv .venv
  fi

  source .venv/bin/activate
  log_info "Virtual environment activated"

  python3 -m pip install --upgrade pip setuptools wheel
  log_info "Upgraded pip, setuptools, wheel"
}

# Install dependencies
install_deps() {
  log_info "Installing project dependencies..."

  if [ -f requirements.txt ]; then
    pip install -r requirements.txt
  fi

  log_info "Dependencies installed"
}

# Setup pre-commit hooks
setup_precommit() {
  log_info "Setting up pre-commit hooks..."

  pip install pre-commit
  pre-commit install
  pre-commit run --all-files 2>/dev/null || true

  log_info "Pre-commit hooks configured"
}

# Install linting/testing tools
install_dev_tools() {
  log_info "Installing development tools..."

  pip install \
    black \
    flake8 \
    mypy \
    isort \
    pylint \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-timeout \
    safety \
    bandit \
    pip-audit

  log_info "Development tools installed"
}

# Setup Drone CI locally
setup_drone() {
  read -p "Do you want to set up Drone CI locally? (y/n) " -n 1 -r
  echo

  if [[ $REPLY =~ ^[Yy]$ ]]; then
    if ! command -v docker &> /dev/null; then
      log_error "Docker is required for Drone CI setup"
      return 1
    fi

    log_info "Starting Drone CI services..."

    # Generate RPC secret
    RPC_SECRET=$(openssl rand -hex 16)
    echo "DRONE_RPC_SECRET=$RPC_SECRET" > .env.drone

    docker-compose -f docker-compose.drone.yml up -d

    log_info "Drone CI started at http://localhost:8080"
    log_warn "Save this RPC secret: $RPC_SECRET"
  fi
}

# Run tests
run_tests() {
  log_info "Running tests..."

  pytest -v --tb=short --cov=. --cov-report=html

  log_info "Tests passed!"
  log_info "Coverage report: htmlcov/index.html"
}

# Security scan
security_scan() {
  log_info "Running security scan..."

  safety check --json > safety-report.json || true
  bandit -r . -f json -o bandit-report.json || true
  pip-audit --desc > pip-audit-report.json || true

  log_info "Security scan complete. Reports saved."
}

# Main setup flow
main() {
  echo ""
  echo "╔════════════════════════════════════════════╗"
  echo "║   CI/CD Environment Setup                  ║"
  echo "╚════════════════════════════════════════════╝"
  echo ""

  check_python
  setup_venv
  install_deps
  install_dev_tools
  setup_precommit

  # Ask about optional components
  read -p "Run tests immediately? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_tests
  fi

  setup_drone

  echo ""
  echo "╔════════════════════════════════════════════╗"
  echo "║   Setup Complete! 🎉                      ║"
  echo "╚════════════════════════════════════════════╝"
  echo ""
  echo "Next steps:"
  echo "  1. Run tests: pytest -v"
  echo "  2. Format code: black ."
  echo "  3. Check types: mypy ."
  echo "  4. Security scan: safety check"
  echo ""
  echo "For more info, see: CI_CD_GUIDE.md"
  echo ""
}

main "$@"
