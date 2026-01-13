# CI/CD Infrastructure - FRED Prime Legal System

This project implements a **multi-layered, open-source CI/CD pipeline** combining GitHub Actions with optional Drone CI for maximum flexibility and no vendor lock-in.

## Quick Start

### GitHub Actions (Recommended)

GitHub Actions is automatically enabled and requires **zero configuration**:

```bash
# Simply push to a branch
git push origin your-branch

# View results in GitHub: Actions tab
# Workflows run automatically on push and pull request
```

### Drone CI (Optional, Self-Hosted)

For a local, self-hosted CI/CD solution:

```bash
# 1. Run setup script
bash setup-ci.sh

# 2. Select "y" when asked to set up Drone CI
# 3. Access at http://localhost:8080
```

## Architecture

### GitHub Actions Workflows

| Workflow        | Trigger                     | Purpose                                               |
| --------------- | --------------------------- | ----------------------------------------------------- |
| **CI Enhanced** | Push/PR to main, develop    | Multi-OS, multi-Python testing with security scanning |
| **Codex Build** | Push/PR to main, codex/\*\* | Intelligent test triggering based on file changes     |
| **Supreme MBP** | Push/PR to main             | Litigation OS specific tests                          |

### Drone CI Pipeline

Runs in Docker containers (portable, reproducible):

- **Lint stage**: Code quality checks (Black, Flake8, MyPy)
- **Test stages**: Python 3.10, 3.11, 3.12 in parallel
- **Security stage**: Dependency and code vulnerability scanning
- **Build stage**: Package distribution (if main branch)

## File Structure

```
├── .github/workflows/
│   ├── ci-improved.yml         ✨ New: Enhanced GitHub Actions
│   ├── build.yml               ⚡ Improved: Codex Build
│   └── ci.yml                  Supreme MBP CI
│
├── .drone.yml                  ✨ New: Drone CI configuration
├── docker-compose.drone.yml    ✨ New: Docker setup for Drone
├── .pre-commit-config.yaml     ✨ New: Git hooks for quality
├── .bandit                     ✨ New: Security scanning config
│
├── setup-ci.sh                 ✨ New: Automated setup script
├── CI_CD_GUIDE.md              ✨ New: Comprehensive documentation
└── README.md                   This file
```

## Key Improvements Over CircleCI

### ✅ No Vendor Lock-in

- **GitHub Actions**: Native to GitHub, runs on GitHub's servers
- **Drone CI**: Open-source, runs on your own servers or self-hosted
- **Both**: Standard YAML configurations, easy to migrate

### ✅ Enhanced Testing

- **Multi-platform**: Linux, macOS, Windows in GitHub Actions
- **Multi-Python**: Test against Python 3.10, 3.11, 3.12
- **Parallel execution**: Tests run in parallel for faster feedback
- **Matrix strategy**: Comprehensive coverage combinations

### ✅ Security Built-in

- **Dependency scanning**: Safety, pip-audit
- **Code analysis**: Bandit, CodeQL
- **Secret detection**: Automated detection of hardcoded secrets
- **Scheduled audits**: Weekly security scans

### ✅ Cost-Effective

- **GitHub Actions**: Free for public repos (2,000 min/month for private)
- **Drone CI**: Free, open-source (self-hosted)
- **No per-minute charges** like CircleCI

### ✅ Developer Experience

- **Local testing**: Run `act` to test workflows locally
- **Pre-commit hooks**: Catch issues before pushing
- **Quick feedback**: Test results in 2-3 minutes
- **Easy debugging**: Full logs accessible

## Setup Instructions

### Automatic Setup

```bash
# Run the setup script (interactive)
bash setup-ci.sh

# Choose options when prompted:
# - Virtual environment creation
# - Dependency installation
# - Pre-commit hook setup
# - Optional: Drone CI setup
```

### Manual Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if exists

# 3. Setup pre-commit hooks
pip install pre-commit
pre-commit install

# 4. Run tests locally
pytest -v --cov=.

# 5. Format code
black .
isort .

# 6. Check types
mypy . --ignore-missing-imports

# 7. Security scan
safety check
bandit -r .
```

## Workflow Execution Flow

### GitHub Actions

```
Push/PR to main/develop
         ↓
GitHub Actions triggered
         ↓
┌─────────────────────────┐
│ Parallel Jobs:          │
│ ├─ Lint                 │
│ ├─ Test (3x Python)     │
│ ├─ Security             │
│ └─ Build (on push)      │
└─────────────────────────┘
         ↓
Results → GitHub PR/Status
         ↓
Artifacts uploaded (coverage, reports)
```

### Drone CI (Local)

```
Git Push
   ↓
Webhook → Drone Server
   ↓
Docker Runner pulls pipeline
   ↓
┌──────────────────┐
│ Sequential/Parallel Steps:
│ ├─ Lint
│ ├─ Test 3.10
│ ├─ Test 3.11 ┐
│ ├─ Test 3.12 ├─ (Can be parallel)
│ ├─ Security  ┘
│ └─ Build
└──────────────────┘
   ↓
Results → Drone Dashboard
   ↓
Artifacts in volume
```

## Detailed Features

### Code Quality (Lint Stage)

Automatically checks:

- **Black**: Code formatting
- **Flake8**: Style and error detection
- **MyPy**: Type checking
- **isort**: Import sorting
- **Pylint**: Advanced code analysis

### Testing (Test Stage)

- **Pytest**: Framework with coverage
- **pytest-xdist**: Parallel test execution
- **pytest-timeout**: Prevent hanging tests
- **pytest-cov**: Coverage reporting
- **Codecov integration**: Track coverage over time

### Security (Security Stage)

- **Safety**: Vulnerable dependency check
- **pip-audit**: Alternative dependency scanner
- **Bandit**: Python security linting
- **GitHub CodeQL**: Advanced static analysis (optional)

### Artifacts & Reports

Generated after successful runs:

- **Coverage reports**: HTML and XML formats
- **Security reports**: JSON for integration
- **Build artifacts**: Python packages (wheel, sdist)
- **Test results**: TAP, JUnit XML formats

## Local Testing with act

Test GitHub Actions workflows locally before pushing:

```bash
# Install act
brew install act  # macOS
# or from https://github.com/nektos/act

# Run specific job
act -j lint

# Run all jobs
act

# Run with specific Python version
act -e environment.json

# List available jobs
act -l
```

## Using Drone CI

### Start Drone (Docker)

```bash
# With docker-compose
docker-compose -f docker-compose.drone.yml up -d

# Verify running
docker ps | grep drone

# View logs
docker logs fredprime-drone
docker logs fredprime-drone-runner
```

### Access Drone UI

```
http://localhost:8080
```

### Configure Repositories

1. Click "Sync" to sync repositories
2. Click toggle to enable a repository
3. Drone automatically reads `.drone.yml`
4. Push code to trigger pipeline

### Environment Variables

Store sensitive data in Drone settings:

```
Repository Settings → Secrets
├─ PYPI_PASSWORD
├─ GITHUB_TOKEN
└─ etc.
```

Reference in pipeline:

```yaml
steps:
  - name: publish
    environment:
      PYPI_PASSWORD:
        from_secret: PYPI_PASSWORD
```

## Pre-commit Hooks

Automatically run checks before each commit:

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Update hooks
pre-commit autoupdate

# Bypass hooks (if needed)
git commit --no-verify
```

Configured checks:

- Code formatting (Black)
- Import sorting (isort)
- Linting (Flake8)
- Type checking (MyPy)
- Security (Bandit)
- YAML validation
- JSON validation
- Secret detection

## Troubleshooting

### Tests Pass Locally but Fail in CI

**Possible causes:**

- Different Python version (verify matrix in workflow)
- Missing dependencies (check requirements.txt)
- Environment variables not set (use secrets)
- Path separator issues (use os.path.join, not /)

**Solutions:**

```bash
# Test exact CI environment locally
act -P ubuntu-latest=-self-hosted
python3.11 -m pytest  # Match CI Python version

# Check environment
echo $PYTHONPATH
which python
```

### Slow CI Builds

**Optimization tips:**

1. Enable pip caching (automatic in setup-python@v5)
2. Use `pytest -n auto` for parallel tests
3. Split tests into multiple jobs
4. Cache Docker layers (for Drone)

### Drone CI Not Starting

```bash
# Check docker daemon
docker ps

# Restart services
docker-compose -f docker-compose.drone.yml restart

# Check logs
docker-compose -f docker-compose.drone.yml logs -f

# Verify RPC secret set
cat .env.drone
```

### Coverage Not Uploading

```bash
# Verify Codecov token
echo $CODECOV_TOKEN

# Manual upload
pip install codecov
codecov -f coverage.xml
```

## Migration Guide

### From CircleCI

CircleCI → GitHub Actions benefits:

- Native GitHub integration
- No separate platform
- Free for public repos
- Better log viewing

```bash
# Convert CircleCI config (NPX)
npx circleci-to-github-actions .circleci/config.yml
```

### From Travis CI

Similar benefits, plus:

- GitHub Actions is the de facto standard
- Better documentation
- Native to GitHub platform

## Continuous Deployment (Future)

Extend workflows for automatic deployment:

```yaml
- name: Deploy to Production
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  run: |
    # Your deployment commands here
    bash scripts/deploy.sh
```

## Resources

- **GitHub Actions**: https://docs.github.com/en/actions
- **Drone CI**: https://docs.drone.io/
- **act (local testing)**: https://github.com/nektos/act
- **Pre-commit**: https://pre-commit.com/
- **Pytest**: https://docs.pytest.org/

## Contributing

When contributing:

1. **Pre-commit checks pass**: `pre-commit run --all-files`
2. **Tests pass locally**: `pytest -v`
3. **Code formatted**: `black . && isort .`
4. **No type errors**: `mypy .`
5. **Security clean**: `bandit -r .`

## Support

For issues:

1. Check CI logs in GitHub Actions or Drone UI
2. Run `bash setup-ci.sh` to ensure correct setup
3. Review `CI_CD_GUIDE.md` for detailed docs
4. Check GitHub Issues for similar problems

---

**Last Updated**: January 2026
**Status**: ✅ Production Ready
