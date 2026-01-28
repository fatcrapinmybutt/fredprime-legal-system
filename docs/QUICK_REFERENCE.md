# Development Quick Reference

Quick reference guide for common development tasks in FRED Supreme Litigation OS.

## Getting Started (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/fatcrapinmybutt/fredprime-legal-system.git
cd fredprime-legal-system

# 2. Setup development environment
make dev-setup

# 3. Verify setup
make quick-check
```

## Common Commands

### Development

```bash
make install          # Install production dependencies
make install-dev      # Install development dependencies
make install-all      # Install everything

make format           # Auto-format code
make lint             # Run linters
make check            # Run all checks
make test             # Run tests
make test-coverage    # Run with coverage report

make pre-commit       # Run pre-commit hooks manually
make enforce-rulesets # Run custom rulesets

python codex_selftest.py  # Run system self-test
```

### Building & Release

```bash
make build            # Create distribution packages
make clean            # Remove build artifacts
make clean-all        # Clean everything
```

### Documentation

```bash
make docs             # Build documentation
```

## Project Structure

```
fredprime-legal-system/
├── src/               # Core library
├── modules/           # Feature modules
├── cli/               # Command-line interfaces
├── scripts/           # Utility scripts
├── tests/             # Test suite
├── docs/              # Documentation
├── .github/           # GitHub config & workflows
│   └── workflows/     # CI/CD pipelines
├── config/            # Configuration files
├── forms/             # Legal form templates
├── Makefile           # Development tasks
├── pyproject.toml     # Project metadata & dependencies
├── .pre-commit-config.yaml  # Pre-commit hooks
└── README.md          # This file
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature
# or: bugfix/fix-name, docs/update-name
```

### 2. Make Changes

```bash
# Edit files in your editor
# Code is formatted and checked automatically on commit
```

### 3. Test Changes

```bash
make test              # Run all tests
make test-fast        # Run quick tests only
make test-coverage    # Check coverage
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new feature"
# Follows conventional commit format
```

### 5. Push & Create PR

```bash
git push origin feature/your-feature
# Then create PR on GitHub
```

## Code Quality

### System Self-Test

The repository includes a self-test script (`codex_selftest.py`) that verifies system integrity:

```bash
python codex_selftest.py
```

**What it checks:**
- Branch naming conventions (relaxed for non-codex branches)
- Commit message format (relaxed for CI/development branches)
- Manifest file integrity (optional)

**Environment Variables:**
- `CODEX_SKIP_STRICT_CHECKS=true` - Skip branch/commit format checks
- `CODEX_SKIP_HASH_CHECKS=true` - Skip manifest hash verification

The self-test automatically detects non-codex branches (e.g., `copilot/*`, `feature/*`) and enables relaxed mode for easier development.

### Automated Checks

Checks run automatically:
- **Pre-commit hooks** - On every commit
- **GitHub Actions** - On every push/PR
- **Enforcement script** - Can run manually

### Manual Checks

```bash
make format           # Auto-fix formatting
make lint             # Check for issues
make type-check       # Type checking
make security-check   # Security scanning
make check            # Run all checks
```

## Testing

### Run Tests

```bash
make test             # All tests
make test-fast        # Exclude slow tests
make test-coverage    # With coverage report
pytest tests/test_specific.py  # Specific file
pytest tests/test_module.py::test_function  # Specific test
```

### Coverage Requirements

- Target: >80% code coverage
- Critical paths: 100% coverage
- View report: `open htmlcov/index.html`

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Use Python Debugger

```python
import pdb
pdb.set_trace()  # Breakpoint
```

### Debug Commands

- `n` - Next line
- `s` - Step into
- `c` - Continue
- `p var` - Print variable
- `h` - Help

## Troubleshooting

### Reset Local Changes

```bash
# Discard all local changes
git checkout -- .

# Reset to remote state
git reset --hard origin/main

# Clean untracked files
git clean -fd
```

### Rebuild Environment

```bash
# Clear all cache and rebuild
make clean-all
make install-all
make quick-check
```

### Pre-commit Hooks Not Running

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install
```

### Import Errors

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall package
pip install -e . --force-reinstall

# Check imports directly
python -c "import modules"
```

## Resources

### Documentation
- [README.md](README.md) - Project overview
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [ADVANCED_DEVELOPMENT.md](docs/ADVANCED_DEVELOPMENT.md) - Technical guide
- [OPEN_SOURCE_RULESETS.md](docs/OPEN_SOURCE_RULESETS.md) - Ruleset documentation

### Tools Documentation
- [Pre-commit Framework](https://pre-commit.com/)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Flake8 Linter](https://flake8.pycqa.org/)
- [MyPy Type Checker](https://mypy.readthedocs.io/)
- [Bandit Security Tool](https://bandit.readthedocs.io/)

### Conventions
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

## Getting Help

- **Questions** - Open a GitHub Discussion
- **Bugs** - Open a GitHub Issue
- **Ideas** - Start a Discussion
- **Urgent** - Use `urgent` label on Issues

## Tips & Tricks

### Speed Up Development

```bash
# Skip slow tests
make test-fast

# Run only linting (not type checking)
flake8 src/

# Run only specific linter
black --check src/
```

### Better Terminal Output

```bash
# More detailed test output
pytest -vv --tb=long

# Show print statements during tests
pytest -s

# Stop on first failure
pytest -x

# Last failed tests first
pytest --lf
```

### Useful Aliases

Add to your `.bashrc` or `.zshrc`:

```bash
alias m='make'
alias mt='make test'
alias mf='make format'
alias ml='make lint'
alias mc='make check'
alias mt='make test'
```

---

## Next Steps

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
2. Check [ADVANCED_DEVELOPMENT.md](docs/ADVANCED_DEVELOPMENT.md) for technical details
3. Review existing code to understand patterns
4. Start with issues labeled `good first issue`

Happy coding! 🚀
