# CI/CD Configuration Guide

This repository uses multiple CI/CD solutions for comprehensive testing and deployment.

## GitHub Actions (Primary)

GitHub Actions is the primary CI/CD platform, providing:

- **Multi-platform testing** (Linux, macOS, Windows)
- **Python 3.10, 3.11, 3.12 matrix testing**
- **Code quality checks** (linting, formatting, type checking)
- **Security scanning** (dependency vulnerabilities, code analysis)
- **Coverage reporting** (integration with Codecov)

### Workflows

| Workflow    | File                                | Purpose                                            |
| ----------- | ----------------------------------- | -------------------------------------------------- |
| CI Enhanced | `.github/workflows/ci-improved.yml` | Improved matrix testing with security scanning     |
| Codex Build | `.github/workflows/build.yml`       | Intelligent test triggering based on changed files |
| Supreme MBP | `.github/workflows/ci.yml`          | Litigation OS specific testing                     |

### Running GitHub Actions Locally

Use [act](https://github.com/nektos/act) to test workflows locally:

```bash
# Install act
brew install act  # macOS
# or
choco install act  # Windows with Chocolatey

# Run a specific workflow
act -j lint

# Run all jobs
act

# Run with specific Python version
act -e env.json
```

Create `env.json`:

```json
{
  "PYTHON_VERSION": "3.11"
}
```

## Drone CI (Optional, Open-Source)

Drone CI is an open-source alternative that can run locally or on a server:

- **Container-native** (all steps run in Docker)
- **No vendor lock-in** (can be self-hosted)
- **Lightweight** (single binary deployment)
- **Parallel step execution**

### Setup Drone CI Locally

#### Option 1: Docker Compose

```bash
# Start Drone server and runner
docker-compose -f docker-compose.drone.yml up -d

# Access Drone UI
open http://localhost:8080
```

Configuration:

- Set `DRONE_RPC_SECRET` in `.env`:
  ```bash
  DRONE_RPC_SECRET=$(openssl rand -hex 16)
  ```

#### Option 2: Standalone Binary

```bash
# Download latest Drone binary
wget https://github.com/harness/drone/releases/download/v2.16.0/drone_linux_amd64.tar.gz
tar -zxf drone_linux_amd64.tar.gz
sudo install -t /usr/local/bin drone

# Start Drone server
drone server --host localhost:8080 --proto http --rpc-secret $(openssl rand -hex 16)

# Start Drone runner (in another terminal)
drone-runner-docker --host localhost:3000 --proto http
```

#### Option 3: Kubernetes Deployment

```bash
helm repo add drone https://charts.drone.io
helm repo update
helm install drone drone/drone \
  --set ingress.enabled=true \
  --set ingress.hostname=drone.example.com
```

### Configure Drone with GitHub

1. Create OAuth app in GitHub:

   - Settings → Developer settings → OAuth Apps
   - Authorization callback URL: `http://localhost:8080/login/github/callback`

2. Set environment variables:

   ```bash
   DRONE_GITHUB_CLIENT_ID=xxxx
   DRONE_GITHUB_CLIENT_SECRET=yyyy
   ```

3. Activate repository in Drone UI

### Configure Drone with Gitea (Self-Hosted)

```bash
# Set in .env
GITEA_SERVER=http://localhost:3000
GITEA_CLIENT_ID=xxxx
GITEA_CLIENT_SECRET=yyyy
```

## Code Quality Tools

### Local Code Quality Checks

Run before committing:

```bash
# Formatting
black .
isort .

# Linting
flake8 . --max-line-length=120

# Type checking
mypy . --ignore-missing-imports

# Security scanning
bandit -r .
safety check
pip-audit
```

### Pre-commit Hooks

Setup automatic checks before commit:

```bash
pip install pre-commit

# Create .pre-commit-config.yaml (see file)
pre-commit install

# Run manually
pre-commit run --all-files
```

## Security Scanning

### Tools Included

| Tool              | Purpose                        | Config           |
| ----------------- | ------------------------------ | ---------------- |
| **safety**        | Vulnerable dependencies        | `safety check`   |
| **pip-audit**     | Alternative dependency scanner | `pip-audit`      |
| **bandit**        | Python code security           | `.bandit`        |
| **GitHub CodeQL** | Advanced code analysis         | `.github/codeql` |

### Running Security Scans Locally

```bash
# Install tools
pip install safety bandit pip-audit

# Run scans
safety check --json > safety-report.json
bandit -r . -f json > bandit-report.json
pip-audit --desc > pip-audit-report.json
```

### CodeQL Integration

```bash
# Initialize CodeQL database
codeql database create fredprime-codeql-db --language=python

# Run analysis
codeql database analyze fredprime-codeql-db codeql/python-queries --format=sarif-latest > results.sarif
```

## Testing Strategy

### Test Execution

```bash
# Run all tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Parallel execution (faster)
pytest -n auto

# With verbose output
pytest -v --tb=short

# Specific test file/function
pytest tests/test_module.py::test_function -v
```

### Coverage Reports

- **Local**: `pytest --cov=. --cov-report=html` → `htmlcov/index.html`
- **CI**: Uploaded to [Codecov](https://codecov.io)
- **Drone**: Coverage artifacts stored in drone-data

## Performance Optimization

### Caching Strategy

**GitHub Actions:**

```yaml
- uses: actions/setup-python@v5
  with:
    cache: 'pip' # Caches pip dependencies
```

**Drone CI:**

```yaml
volumes:
  - name: pip_cache
    path: /root/.cache/pip
```

### Parallel Testing

```bash
# GitHub Actions (matrix strategy)
strategy:
  matrix:
    python-version: [3.10, 3.11, 3.12]
    os: [ubuntu-latest, macos-latest, windows-latest]

# Local with pytest-xdist
pytest -n auto
```

## Deployment

### Pushing to PyPI

```bash
# Build distribution
python -m build

# Upload to PyPI (requires credentials)
twine upload dist/*
```

### Creating GitHub Releases

Automatically triggered on version tags:

```bash
git tag v1.0.0
git push origin v1.0.0
```

## Troubleshooting

### GitHub Actions Debugging

```bash
# Enable debug logging
export ACTIONS_STEP_DEBUG=true

# Run act with verbose output
act -v
```

### Drone CI Debugging

```bash
# View logs
docker logs fredprime-drone

# View runner logs
docker logs fredprime-drone-runner

# Enable debug mode
DRONE_LOGS_DEBUG=true
```

### Common Issues

| Issue                   | Solution                                            |
| ----------------------- | --------------------------------------------------- |
| Tests timeout           | Increase `timeout-minutes` in workflow              |
| Cache not working       | Clear cache in Actions tab                          |
| Python version mismatch | Use `actions/setup-python@v5` with explicit version |
| Docker socket error     | Mount `/var/run/docker.sock` in Drone runner        |

## Migration Between CI/CD Systems

### From CircleCI to GitHub Actions

```bash
# Convert CircleCI config
npx circleci-to-github-actions .circleci/config.yml
```

### From GitHub Actions to Drone

```bash
# Drone supports similar syntax
# Minimal adaptation needed:
# - GitHub: uses/checkout@v4 → drone: plugins/git
# - GitHub: actions/setup-python → drone: image: python:x.x
```

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Drone CI Documentation](https://docs.drone.io/)
- [Gitea Documentation](https://docs.gitea.io/)
- [Pre-commit Framework](https://pre-commit.com/)
- [pytest Documentation](https://docs.pytest.org/)
- [CodeQL Documentation](https://codeql.github.com/docs/)
