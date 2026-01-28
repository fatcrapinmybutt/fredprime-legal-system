# CircleCI Configuration

This repository includes a comprehensive CircleCI pipeline configuration for continuous integration and testing.

## Overview

The CircleCI pipeline provides:

- **Multi-version Python testing** (Python 3.10, 3.11, 3.12)
- **Code quality checks** (linting, formatting, type checking)
- **Security scanning** (dependency vulnerabilities, code analysis)
- **Test execution with coverage** (pytest with parallel execution)
- **Build and packaging** (distribution package creation)
- **Documentation validation** (markdown and YAML linting)
- **Scheduled security scans** (weekly automatic scans)

## Pipeline Structure

### Jobs

| Job            | Description                                           | Python Versions |
| -------------- | ----------------------------------------------------- | --------------- |
| `lint`         | Code formatting, import sorting, linting, type checks | 3.11            |
| `test`         | Run pytest suite with coverage reporting              | 3.10, 3.11, 3.12|
| `security`     | Dependency and code security scanning                 | 3.11            |
| `build`        | Create distribution packages                          | 3.11            |
| `documentation`| Validate markdown and YAML files                      | N/A (Node)      |

### Workflows

#### Main CI Pipeline (`ci-pipeline`)

Runs on:

- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

Job flow:

```
lint
├── test-python-3.10
├── test-python-3.11
├── test-python-3.12
├── security
└── documentation
    └── build (only on main/develop)
```

#### Weekly Security Scan (`weekly-security`)

Runs every Sunday at 2 AM UTC on the `main` branch for proactive security monitoring.

## Setup Instructions

### 1. Connect Repository to CircleCI

1. Go to [CircleCI](https://circleci.com/)
2. Sign in with your GitHub account
3. Navigate to Projects
4. Find `fatcrapinmybutt/fredprime-legal-system`
5. Click "Set Up Project"
6. Choose "Use Existing Config" (config already in `.circleci/config.yml`)
7. Click "Start Building"

### 2. Configure Environment Variables (Optional)

If you need to upload coverage to Codecov:

1. Go to Project Settings in CircleCI
2. Navigate to Environment Variables
3. Add:
   - `CODECOV_TOKEN`: Your Codecov token (optional, works without for public repos)

### 3. Configure Orbs (Already Included)

The configuration uses these orbs:

- `circleci/python@2.1.1` - Python environment management
- `codecov/codecov@4.0.1` - Coverage reporting

These are public orbs and require no additional setup.

## Local Validation

### Validate Config Syntax

```bash
# Install CircleCI CLI
brew install circleci  # macOS
# or
sudo snap install circleci  # Linux

# Validate configuration
circleci config validate .circleci/config.yml
```

### Test Locally with CircleCI CLI

```bash
# Run a specific job locally
circleci local execute --job lint

# Run the entire workflow (limited support)
circleci local execute
```

Note: Local execution has limitations (no workflows, no orbs, no caching).

## Features

### Dependency Caching

The pipeline caches pip dependencies to speed up builds:

```yaml
- restore_cache:
    keys:
      - deps-v1-py{{ .python-version }}-{{ checksum "requirements.txt" }}
```

### Parallel Testing

Tests run in parallel across multiple Python versions:

- Python 3.10
- Python 3.11
- Python 3.12

### Code Coverage

Coverage reports are:

- Generated with pytest-cov
- Uploaded to Codecov
- Stored as artifacts (HTML reports)

### Security Scanning

Three security tools run automatically:

- **Safety**: Checks for known vulnerabilities in dependencies
- **Pip-audit**: Alternative dependency vulnerability scanner
- **Bandit**: Static code analysis for security issues

### Artifact Storage

The pipeline stores:

- Test results (JUnit XML)
- Coverage reports (HTML)
- Security scan reports (JSON)
- Build distributions (wheels, sdists)

## Comparing with GitHub Actions

| Feature              | GitHub Actions        | CircleCI                 |
| -------------------- | --------------------- | ------------------------ |
| **Cost**             | Free for public repos | Free tier available      |
| **Concurrency**      | 20 jobs (free)        | 1 container (free tier)  |
| **Docker Support**   | Yes                   | Native                   |
| **Caching**          | Built-in              | Built-in                 |
| **Matrix Builds**    | Yes (built-in)        | Yes (via parameters)     |
| **Local Testing**    | act tool              | CircleCI CLI             |
| **Configuration**    | YAML (workflows)      | YAML (jobs/workflows)    |

## Troubleshooting

### Common Issues

#### Job Times Out

Increase `no_output_timeout` in the job step:

```yaml
- run:
    command: pytest
    no_output_timeout: 20m
```

#### Dependency Installation Fails

Check that `requirements.txt` is valid:

```bash
pip install -r requirements.txt
```

#### Cache Not Working

Clear CircleCI cache:

1. Go to Project Settings
2. Advanced Settings
3. Click "Clear Cache"

#### Tests Fail on CircleCI but Pass Locally

Common causes:

- Different Python versions
- Missing environment variables
- Timing issues (increase timeouts)
- File system differences

### Getting Help

- [CircleCI Documentation](https://circleci.com/docs/)
- [CircleCI Community Forum](https://discuss.circleci.com/)
- [CircleCI Status Page](https://status.circleci.com/)

## Migration from Other CI Systems

### From GitHub Actions

The CircleCI config mirrors the GitHub Actions workflows:

- `.github/workflows/ci-improved.yml` → `.circleci/config.yml`
- Similar job structure and commands
- Compatible with the same test suite

Both systems can run in parallel without conflicts.

### From Travis CI or Jenkins

CircleCI uses Docker-based executors, which may require adjusting:

- Container images (use `cimg/*` CircleCI images)
- Environment setup
- Caching strategies

## Best Practices

1. **Keep jobs focused**: Each job should have a single responsibility
2. **Use caching**: Cache dependencies to speed up builds
3. **Fail fast**: Use `fail-fast: false` in test matrix for all results
4. **Store artifacts**: Save test results and logs for debugging
5. **Monitor performance**: Check job execution times regularly
6. **Update orbs**: Keep orb versions current for security fixes

## Next Steps

1. **Enable CircleCI** for this repository
2. **Monitor first builds** to ensure everything works
3. **Review coverage reports** uploaded to Codecov
4. **Check security scans** in the artifacts
5. **Customize workflows** based on your needs

## Support

For issues specific to this configuration:

- Open an issue in the repository
- Reference the CircleCI build URL
- Include relevant logs

For general CircleCI issues:

- Check [CircleCI Docs](https://circleci.com/docs/)
- Visit [CircleCI Support](https://support.circleci.com/)
