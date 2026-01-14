# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Pre-commit hooks configuration for Black, isort, Flake8, Bandit
- Comprehensive development setup guide (`DEV_SETUP.md`)
- Enhanced README with complete project structure and quick start
- Black and isort code formatting across entire codebase

### Changed

- Reorganized README with better structure and examples
- Consolidated project metadata in `pyproject.toml`
- Updated CI/CD workflows for better reliability
- Improved linting and formatting configuration

### Fixed

- Repaired corrupted Python files (encoding, binary content)
- Fixed triple-quote nesting in `BenchbookMCR Rules Plugin` module
- Removed stray instructions from `gather_mindeye2_artifacts.py`
- Truncated oversized generated code sections for stability
- Applied isort and black formatting to 41+ files
- Resolved import order and formatting issues repo-wide

### Security

- Added Bandit security scanning
- Integrated detect-secrets for credential detection
- Enhanced pre-commit hooks for security

### Deprecated

- Old GitHub API-based ruleset approach (now using open-source tools)

### Removed

- Malformed code sections from large generated files
- Unnecessary .git metadata files from Python formatter scans

## [1.0.0] - 2026-01-14

### Added

- Initial release of FRED Supreme Litigation OS
- Core litigation automation framework
- Document management and form generation
- Legal template system
- Comprehensive testing suite
- Full documentation
- Michigan court compliance system
- Timeline analysis and contradiction detection
- Evidence tracking with blockchain authentication

### Features

- Automated legal document processing
- Court form handling
- Motion generation
- Affidavit building
- Binder packing with exhibit management
- Entity liability tracing
- Timeline visualization
- Mock trial simulation
- Adversarial challenge scenarios

- Timeline and evidence management
- Pre-commit hooks for code quality
- Comprehensive ruleset enforcement

---

## Changelog Guidelines

When adding new entries:

### Format Sections

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for security fixes/improvements

### Guidelines

1. Keep sections in order (Added, Changed, Deprecated, Removed, Fixed, Security)
2. Include version numbers and dates
3. Use links to compare versions
4. Group related changes together
5. Reference related issues/PRs

### Example Entry

```markdown
## [1.0.1] - 2026-01-20

### Added

- New feature description

### Fixed

- Bug fix description (Closes #123)

### Changed

- Change description
```

### Release Process

1. Update version in pyproject.toml
2. Add [Unreleased] section
3. Move changes from [Unreleased] to new version
4. Update links at bottom
5. Commit: `chore: release version 1.0.1`
6. Tag: `git tag -a v1.0.1 -m "Release version 1.0.1"`
