# Contributing to FRED Supreme Litigation OS

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

Be respectful, inclusive, and professional. Harassment or abuse will not be tolerated.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fredprime-legal-system.git
   cd fredprime-legal-system
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/fatcrapinmybutt/fredprime-legal-system.git
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites
- Python 3.10+
- Git
- pip or conda

### Setup Steps

```bash
# 1. Install development dependencies
make install-all

# OR manually:
pip install -e ".[dev,docs]"

# 2. Install pre-commit hooks
pre-commit install

# 3. Verify setup
make quick-check
```

## Making Changes

### Creating a Feature Branch

```bash
# Update main first
git checkout main
git pull upstream main

# Create feature branch with conventional name
git checkout -b feature/add-new-feature
git checkout -b bugfix/fix-issue-123
git checkout -b docs/improve-documentation
git checkout -b refactor/clean-up-module
```

### Branch Naming Conventions

- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `docs/*` - Documentation updates
- `refactor/*` - Code refactoring
- `test/*` - Test additions/improvements
- `chore/*` - Maintenance tasks
- `hotfix/*` - Critical production fixes

### Making Your Changes

```bash
# Make your changes in your editor
# Run formatters and linters before committing
make format
make lint

# Stage and commit (see Commit Guidelines below)
git add .
git commit -m "feat: your meaningful commit message"
```

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that don't affect code meaning (formatting, missing semicolons, etc.)
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Code change that improves performance
- **test**: Adding or updating tests
- **chore**: Changes to build process, dependencies, or tooling

### Scope (Optional)

Area of the codebase affected (e.g., `api`, `cli`, `core`, `docs`)

### Subject

- Use imperative mood ("add" not "added" or "adds")
- Don't capitalize first letter
- No period (.) at the end
- Limit to 50 characters

### Body (Optional)

- Explain what and why, not how
- Wrap at 72 characters
- Separate from subject with blank line

### Footer (Optional)

- Reference issues: `Closes #123`, `Fixes #456`
- Note breaking changes: `BREAKING CHANGE: description`

### Examples

```
feat(core): add document validation

Add schema-based validation for legal documents
to ensure compliance with court requirements.

Closes #123
```

```
fix(cli): resolve argument parsing issue

Command-line arguments were not being parsed
correctly when using multiple flags together.

Fixes #456
```

## Pull Request Process

### Before Submitting

1. **Update from upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   make check
   make test-coverage
   ```

3. **Ensure pre-commit passes**:
   ```bash
   pre-commit run --all-files
   ```

### Submitting a PR

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub
   - Use a clear title following conventions
   - Link related issues
   - Describe what changed and why
   - Include any breaking changes

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Related Issues
   Closes #123

   ## Testing
   - [ ] Tests pass
   - [ ] Coverage maintained
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Comments added for complex logic
   - [ ] Documentation updated
   - [ ] No new warnings generated
   ```

### PR Requirements

- ✓ All tests pass
- ✓ Code coverage maintained (>80%)
- ✓ No linting errors
- ✓ Code is formatted (Black)
- ✓ Commits follow conventions
- ✓ Documentation updated (if applicable)

## Code Style

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these tools:

- **Black** for code formatting
- **isort** for import organization
- **Flake8** for linting

### Format Code

```bash
# Automatic formatting
make format

# Manual verification
make lint
```

### Type Hints

Use type hints in function signatures:

```python
def process_document(path: str, validate: bool = True) -> dict:
    """Process a legal document.
    
    Args:
        path: Path to the document file
        validate: Whether to validate the document
        
    Returns:
        Dictionary containing processed document data
    """
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_deadline(start_date: str, days: int) -> str:
    """Calculate a deadline date from start date and number of days.
    
    Args:
        start_date: Start date in ISO format (YYYY-MM-DD)
        days: Number of days to add
        
    Returns:
        Deadline date in ISO format (YYYY-MM-DD)
        
    Raises:
        ValueError: If start_date is not in valid format
        
    Example:
        >>> calculate_deadline("2024-01-01", 30)
        "2024-01-31"
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
pytest tests/test_module.py

# Run specific test
pytest tests/test_module.py::test_function

# Run fast tests only (skip slow tests)
make test-fast
```

### Writing Tests

```python
import pytest
from module import function

class TestFunctionality:
    """Test suite for feature functionality."""
    
    def test_basic_case(self):
        """Test basic functionality."""
        result = function(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge cases."""
        result = function(edge_case_data)
        assert result is not None
    
    @pytest.mark.slow
    def test_performance(self):
        """Test performance (marked as slow)."""
        # Performance test code
        pass
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function(invalid_data)
```

### Coverage Requirements

- Target: >80% code coverage
- Critical paths: 100% coverage
- Run: `make test-coverage`

## Documentation

### Updating Documentation

1. **README.md** - Project overview and quick start
2. **docs/** - Detailed documentation
3. **Docstrings** - Function and module documentation
4. **CONTRIBUTING.md** - Contribution guidelines
5. **CHANGELOG.md** - Version history

### Writing Documentation

- Use clear, concise language
- Include examples
- Keep markdown well-formatted
- Link to related sections
- Update table of contents

### Documentation Checklist

- [ ] README updated if needed
- [ ] Function/class docstrings added
- [ ] Examples included
- [ ] Related documentation linked
- [ ] Markdown linting passes

## Review Process

### What to Expect

1. **Automated Checks**: CI pipeline runs tests and linters
2. **Code Review**: Maintainers review code quality and approach
3. **Feedback**: Constructive suggestions for improvement
4. **Approval**: PRs merged after approval and all checks pass

### Addressing Feedback

- Be respectful and professional
- Discuss concerns if you disagree
- Make requested changes
- Push updates to the same branch
- PR will update automatically

## Getting Help

- **Questions**: Open a Discussion on GitHub
- **Bugs**: Open an Issue with reproduction steps
- **Ideas**: Start a Discussion first
- **Urgent**: Use GitHub Issues with `urgent` label

## Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [pytest Documentation](https://docs.pytest.org/)
- [Pre-commit Framework](https://pre-commit.com/)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to FRED Supreme Litigation OS! 🎉
