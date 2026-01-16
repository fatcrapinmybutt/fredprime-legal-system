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
  - [Legal Compliance Requirements](#legal-compliance-requirements)
  - [GitHub Copilot Instructions](#github-copilot-instructions)
  - [Python Style](#python-style)
  - [Type Hints](#type-hints)
  - [Docstrings](#docstrings)
- [Testing](#testing)
- [Documentation](#documentation)
- [Review Process](#review-process)
- [Best Practices](#best-practices)
- [Getting Help](#getting-help)
- [Resources](#resources)
- [License](#license)

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
# Option 1: Recommended - Complete development setup
make dev-setup          # Installs all dependencies + pre-commit hooks

# Option 2: Manual setup
make install-all        # Install all dependencies (production + dev + docs)
make git-hooks          # Install pre-commit hooks

# Option 3: Production only
make install            # Install production dependencies only

# Option 4: Development dependencies only
make install-dev        # Install dev dependencies
pip install -e ".[dev]"

# Verify installation
make quick-check        # Quick validation: format + lint
make check              # Full validation: lint + type-check + security + tests
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
   make check              # Runs lint, type-check, security-check, and test
   make test-coverage      # Verify coverage requirements
   ```

3. **Ensure pre-commit passes**:
   ```bash
   pre-commit run --all-files
   ```

4. **Optional: Simulate full CI pipeline**:
   ```bash
   make ci                 # Full CI simulation: clean, install-all, check, test-coverage
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
- ✓ Code coverage maintained (minimum >70% for new code)
- ✓ No linting errors (flake8, isort, black checks pass)
- ✓ Code is formatted (Black + isort)
- ✓ Type checking passes (mypy)
- ✓ Security checks pass (bandit)
- ✓ Commits follow Conventional Commits specification
- ✓ Documentation updated (if applicable)
- ✓ Pre-commit hooks pass

## Code Style

### Legal Compliance Requirements

⚖️ **This repository contains legal system automation. Special requirements apply:**

#### Michigan Court Rules (MCR) Compliance

When contributing code that generates court documents:

- **Signature blocks** must comply with MCR 1.109(D)(3)
- **Court forms** must match official Michigan court form templates
- **Filing deadlines** must be calculated correctly (calendar days vs. business days)
- **Service requirements** must account for MCR service rules

#### Evidence and Chain-of-Custody

When contributing to evidence handling modules:

- **Chain-of-custody** must be maintained and auditable
- **Blockchain logging** must be preserved for evidence timestamping
- **Forensic integrity** checks must not be bypassed
- **Audit trails** are mandatory for all evidence operations

#### Security and Confidentiality

- **Never commit** case information, client data, or sensitive legal materials
- **Never log** confidential information in plain text
- **Always validate** and sanitize user inputs for court document generation
- **Prevent injection attacks** in document templates and form generation

See `COPILOT_AGENT.md` for detailed enforcement rules.

---

### GitHub Copilot Instructions

If you're using GitHub Copilot, our repository includes production-grade agent instructions that enforce:

- Production-quality code (no stubs, TODOs, or placeholders)
- Fail-closed engineering with explicit error handling
- Security-first design with legal compliance requirements
- Deterministic and reproducible solutions

See `.github/copilot-instructions.md` or `COPILOT_AGENT.md` for complete details.

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these tools:

- **Black** - Automatic code formatting (120 char line length)
- **isort** - Import statement organization
- **Flake8** - Linting and style enforcement
- **mypy** - Static type checking
- **Bandit** - Security vulnerability scanning

**Running Quality Checks:**

```bash
make lint               # Flake8 + isort + Black checks
make type-check         # mypy type checking
make security-check     # Bandit security analysis
make check              # All checks + tests (recommended before PR)
```

### Format Code

```bash
# Automatic formatting (recommended)
make format             # Runs isort + Black on all code

# Check formatting without modifying files
make lint               # Runs flake8, isort --check, and black --check

# Individual formatters
isort src modules cli scripts tests         # Sort imports
black src modules cli scripts tests         # Format code
```

### Type Hints

Use type hints for all public functions and methods following PEP 484:

```python
from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DocumentResult:
    """Result of document processing operation."""
    success: bool
    path: Optional[Path] = None
    errors: List[str] = None
    
def process_document(
    path: Path, 
    validate: bool = True,
    output_dir: Optional[Path] = None
) -> Union[DocumentResult, Dict[str, str]]:
    """Process a legal document with validation.
    
    Args:
        path: Path to the document file (must exist)
        validate: Whether to validate MCR compliance (default: True)
        output_dir: Optional output directory for processed document
        
    Returns:
        DocumentResult object on success, or error dictionary on failure
        
    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If document format is invalid
        
    Example:
        >>> result = process_document(Path("motion.docx"))
        >>> if result.success:
        ...     print(f"Processed: {result.path}")
    """
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    
    # Implementation
    return DocumentResult(success=True, path=output_path)
```

**Type Hint Best Practices:**
- Use `Optional[T]` for values that can be `None`
- Use `Union[A, B]` for multiple possible types
- Use `List[T]`, `Dict[K, V]`, `Tuple[T, ...]` for collections
- Use `Path` from pathlib instead of `str` for file paths
- Use dataclasses for structured return types
- Import from `typing` module for Python 3.10 compatibility

### Docstrings

Use Google-style docstrings for consistency with the codebase:

```python
from datetime import datetime, timedelta
from typing import Optional

def calculate_deadline(
    start_date: str, 
    days: int,
    business_days: bool = False
) -> str:
    """Calculate a legal deadline from start date and number of days.
    
    Calculates court deadlines accounting for calendar days or business days
    as required by Michigan Court Rules. Weekends and court holidays are
    excluded when business_days=True.
    
    Args:
        start_date: Start date in ISO format (YYYY-MM-DD)
        days: Number of days to add (must be positive)
        business_days: If True, count only business days (default: False)
        
    Returns:
        Deadline date in ISO format (YYYY-MM-DD)
        
    Raises:
        ValueError: If start_date is not in valid ISO format
        ValueError: If days is not positive
        
    Example:
        >>> # Calendar days (includes weekends)
        >>> calculate_deadline("2024-01-01", 30)
        "2024-01-31"
        
        >>> # Business days only (excludes weekends)
        >>> calculate_deadline("2024-01-01", 21, business_days=True)
        "2024-01-30"
        
    Note:
        This function is critical for court filing deadlines. Any changes
        must be reviewed for MCR compliance.
    """
    # Validate inputs
    try:
        start = datetime.fromisoformat(start_date)
    except ValueError:
        raise ValueError(f"Invalid date format: {start_date}. Use YYYY-MM-DD")
    
    if days <= 0:
        raise ValueError(f"Days must be positive, got: {days}")
    
    # Implementation
    if business_days:
        # Count only business days
        deadline = start
        days_added = 0
        while days_added < days:
            deadline += timedelta(days=1)
            if deadline.weekday() < 5:  # Monday = 0, Sunday = 6
                days_added += 1
    else:
        # Calendar days
        deadline = start + timedelta(days=days)
    
    return deadline.date().isoformat()
```

**Docstring Requirements:**
- **One-line summary**: Brief description (imperative mood)
- **Extended description**: Detailed explanation of functionality
- **Args**: Document all parameters with types and constraints
- **Returns**: Describe return value and type
- **Raises**: Document all exceptions that can be raised
- **Example**: Include usage examples for complex functions
- **Note/Warning**: Add legal compliance or security notes when applicable

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make test-coverage

# Run fast tests only (skip slow tests marked as @pytest.mark.slow)
make test-fast

# Run verbose tests with detailed output
make test-verbose

# Run specific test file
pytest tests/test_module.py

# Run specific test
pytest tests/test_module.py::test_function

# Run tests by marker
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

### Writing Tests

Use pytest with appropriate markers and comprehensive assertions:

```python
import pytest
from pathlib import Path
from modules.document_validator import validate_mcr_compliance

class TestCourtDocumentGeneration:
    """Test suite for court document generation and MCR compliance."""
    
    @pytest.fixture
    def sample_motion(self):
        """Provide sample motion for testing."""
        return {
            "title": "Motion for Summary Disposition",
            "case_number": "2024-001234-DM",
            "signature_block": "Test Attorney\nP12345"
        }
    
    @pytest.mark.unit
    def test_signature_block_compliance(self, sample_motion):
        """Test MCR 1.109(D)(3) signature block compliance."""
        result = validate_mcr_compliance(sample_motion)
        assert result.is_valid
        assert result.signature_compliant
        assert "P12345" in result.signature_block
    
    @pytest.mark.integration
    def test_document_generation_end_to_end(self, sample_motion):
        """Test complete document generation workflow."""
        from modules.motion_generator import generate_motion
        
        motion = generate_motion(sample_motion)
        assert motion is not None
        assert Path(motion.output_path).exists()
        
        # Verify court rules compliance
        compliance = validate_mcr_compliance(motion)
        assert compliance.is_valid
    
    @pytest.mark.slow
    def test_bulk_document_processing(self):
        """Test batch processing performance (marked as slow)."""
        # Performance test for processing multiple documents
        pass
    
    def test_invalid_signature_block_rejected(self):
        """Test that invalid signature blocks are rejected."""
        invalid_motion = {
            "signature_block": "Missing bar number"
        }
        
        with pytest.raises(ValueError, match="signature block.*required"):
            validate_mcr_compliance(invalid_motion)
    
    @pytest.mark.unit
    def test_chain_of_custody_preserved(self):
        """Test evidence chain-of-custody tracking."""
        from forensic.evidence_tracker import EvidenceChain
        
        chain = EvidenceChain()
        chain.add_event("COLLECTED", "2024-01-01", "Officer Smith")
        chain.add_event("PROCESSED", "2024-01-02", "Lab Tech Jones")
        
        assert chain.is_valid()
        assert len(chain.events) == 2
        assert chain.verify_integrity()
```

**Testing Best Practices:**
- Test both success and failure cases
- Use descriptive test names explaining what is tested
- Mark slow tests with `@pytest.mark.slow`
- Test legal compliance requirements explicitly
- Verify error handling and edge cases
- Test chain-of-custody for evidence code
- Use fixtures for reusable test data

### Coverage Requirements

- Minimum: >70% code coverage for new code
- Target: >80% overall code coverage
- Critical paths: 100% coverage required
- Run: `make test-coverage`

## Documentation

### Repository Architecture Overview

Understanding the codebase structure helps you contribute effectively:

```
fredprime-legal-system/
├── core/                  # System-level logic (build_system, codex_brain, patch_manager)
├── modules/               # Reusable litigation components (affidavit_builder, motion_generator)
├── gui/                   # User interface components
├── cli/                   # Command-line interfaces
├── src/                   # Source code modules
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── docs/                  # Documentation
├── contradictions/        # Timeline and statement analysis
├── forms/                 # Court form templates
├── binder/                # Binder generation and exhibit management
└── foia/                  # FOIA request automation
```

**Key Design Patterns:**
- Plugin architecture for module hot-swapping
- Builder pattern for document generation
- Chain-of-custody pattern for evidence tracking
- Observer pattern for system updates

See `COPILOT_AGENT.md` section 13.3 for detailed architecture patterns.

---

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

1. **Automated Checks**: CI pipeline runs tests, linters, type checking, and security scans
2. **Code Review**: Maintainers review code quality, approach, and legal compliance
3. **Feedback**: Constructive suggestions for improvement
4. **Testing Verification**: Ensure tests are comprehensive and pass
5. **Documentation Review**: Check that documentation is updated
6. **Approval**: PRs merged after approval and all checks pass

### Review Criteria

**Code Quality:**
- [ ] Follows PEP 8 and repository style guidelines
- [ ] Uses appropriate design patterns
- [ ] No code duplication (DRY principle)
- [ ] Clear variable and function names
- [ ] Appropriate error handling

**Testing:**
- [ ] New code has corresponding tests
- [ ] Tests cover success and failure cases
- [ ] Tests include edge cases
- [ ] Coverage requirements met (>70% for new code)

**Legal Compliance:**
- [ ] Court document generation follows MCR rules
- [ ] Evidence handling preserves chain-of-custody
- [ ] No confidential data in code or tests
- [ ] Security vulnerabilities addressed

**Documentation:**
- [ ] Public APIs have docstrings
- [ ] Complex logic is commented
- [ ] README or docs updated if needed
- [ ] CHANGELOG.md updated for significant changes

### Addressing Feedback

- Be respectful and professional
- Discuss concerns if you disagree
- Make requested changes
- Push updates to the same branch
- PR will update automatically

## Best Practices

### General Development

✅ **DO:**
- Write clear, self-documenting code
- Keep functions small and focused (single responsibility)
- Use meaningful variable and function names
- Write tests before or alongside code (TDD recommended)
- Commit small, logical changes
- Update documentation as you code
- Run `make format` before committing
- Run `make check` before pushing

❌ **DON'T:**
- Commit commented-out code (use git history instead)
- Hard-code configuration values
- Ignore linting or type errors
- Skip writing tests
- Commit secrets, API keys, or sensitive data
- Make large, multi-purpose commits
- Push without running local checks

### Legal System Development

✅ **DO:**
- Validate all court document templates against official forms
- Test deadline calculations thoroughly (calendar vs business days)
- Maintain audit trails for evidence operations
- Use proper error handling for court filing operations
- Document legal compliance requirements in code
- Review MCR rules when modifying court document generation

❌ **DON'T:**
- Modify evidence chain-of-custody without preserving integrity
- Skip validation for court document generation
- Hard-code court forms or legal text
- Make assumptions about legal procedures
- Test with real case data (use synthetic test data)

### Security Guidelines

✅ **DO:**
- Validate and sanitize all user inputs
- Use parameterized queries for database operations
- Keep dependencies updated (review Dependabot PRs)
- Log security-relevant events
- Use environment variables for sensitive configuration
- Review code for injection vulnerabilities

❌ **DON'T:**
- Log sensitive information (credentials, PII, case details)
- Trust user input without validation
- Use `eval()` or `exec()` on user data
- Disable security features for convenience
- Store secrets in code or version control

### Performance Considerations

✅ **DO:**
- Profile code before optimizing
- Use appropriate data structures
- Mark slow tests with `@pytest.mark.slow`
- Document performance requirements
- Consider scalability for batch operations

❌ **DON'T:**
- Premature optimization
- Load large files entirely into memory
- Block on synchronous I/O unnecessarily
- Ignore time complexity for large datasets

## Getting Help

### Common Issues

<details>
<summary><strong>Installation Problems</strong></summary>

```bash
# If pip install fails, try upgrading pip first
python -m pip install --upgrade pip

# If dependencies conflict, try creating a fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
make install-all
```
</details>

<details>
<summary><strong>Pre-commit Hook Failures</strong></summary>

```bash
# If pre-commit hooks fail, run formatters first
make format

# Then try pre-commit again
pre-commit run --all-files

# If hooks are not installed
make git-hooks
```
</details>

<details>
<summary><strong>Test Failures</strong></summary>

```bash
# Run tests with verbose output to see details
make test-verbose

# Run specific failing test
pytest tests/test_module.py::test_name -vv

# Clear cache and retry
make clean-cache
make test
```
</details>

<details>
<summary><strong>Type Checking Errors</strong></summary>

```bash
# Type checking can be strict. For development:
# - Add type hints gradually
# - Use # type: ignore for external library issues
# - Check mypy configuration in pyproject.toml
make type-check
```
</details>

### Support Channels

- **Questions**: Open a Discussion on GitHub
- **Bugs**: Open an Issue with reproduction steps
- **Ideas**: Start a Discussion first
- **Urgent**: Use GitHub Issues with `urgent` label
- **Security**: Email maintainers directly (see SECURITY.md if available)

## Resources

### External Documentation
- [Conventional Commits](https://www.conventionalcommits.org/) - Commit message standards
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/) - Python style guide
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) - Docstring format
- [pytest Documentation](https://docs.pytest.org/) - Testing framework
- [Pre-commit Framework](https://pre-commit.com/) - Git hooks management
- [Black Code Style](https://black.readthedocs.io/) - Opinionated formatter
- [Type Hints (PEP 484)](https://www.python.org/dev/peps/pep-0484/) - Python typing

### Repository-Specific Documentation
- [COPILOT_AGENT.md](COPILOT_AGENT.md) - AI agent instructions and repository standards
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - GitHub Copilot integration guide
- [docs/COPILOT_USAGE.md](docs/COPILOT_USAGE.md) - How to use Copilot in this repository
- [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - Quick command reference
- [docs/ADVANCED_DEVELOPMENT.md](docs/ADVANCED_DEVELOPMENT.md) - Advanced development patterns
- [CI_CD_GUIDE.md](CI_CD_GUIDE.md) - Continuous integration setup
- [CHANGELOG.md](CHANGELOG.md) - Version history and release notes

### Legal and Court Resources
- [Michigan Court Rules](https://courts.michigan.gov/rules) - MCR reference
- [Michigan Court Forms](https://courts.michigan.gov/forms) - Official court forms
- [MiFILE Information](https://mifile.courts.michigan.gov/) - Electronic filing system

### Development Tools
- [Python Packaging Guide](https://packaging.python.org/) - Python packaging
- [GitHub Flow](https://guides.github.com/introduction/flow/) - GitHub workflow
- [Semantic Versioning](https://semver.org/) - Version numbering scheme

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to FRED Supreme Litigation OS! 🎉
