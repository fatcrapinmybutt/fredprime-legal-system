# Open Source Repository Ruleset System

This repository uses **100% open-source, free tools** for enforcing code quality and
repository standards without requiring GitHub API access or admin permissions.

## Quick Start

### 1. Install Dependencies

```bash
pip install pre-commit
pre-commit install
```

### 2. Run Rulesets

#### Automatically (on each commit)

```bash
# Pre-commit hooks run automatically before each commit
# To test all files:
pre-commit run --all-files
```

#### Manually (anytime)

```bash
bash scripts/enforce_rulesets.sh
```

#### Strict Mode (fails on errors)

```bash
ENFORCE_STRICT=1 bash scripts/enforce_rulesets.sh
```

## Ruleset Components

### Pre-Commit Hooks

Configured in [.pre-commit-config.yaml](.pre-commit-config.yaml), includes:

| Tool | Purpose | Link |
| ------ | --------- | ------ |
| **Black** | Code formatting | <https://github.com/psf/black> |
| **isort** | Import organization | <https://github.com/PyCQA/isort> |
| **Flake8** | Code linting | <https://github.com/PyCQA/flake8> |
| **MyPy** | Type checking | <https://github.com/python/mypy> |
| **Bandit** | Security scanning | <https://github.com/PyCQA/bandit> |
| **Detect Secrets** | Secrets detection | <https://github.com/Yelp/detect-secrets> |
| **Markdownlint** | Markdown validation | <https://github.com/igorshubovych/markdownlint-cli> |
| **Actionlint** | GitHub Actions validation | <https://github.com/rhysd/actionlint> |

### Custom Enforcement Script

[scripts/enforce_rulesets.sh](../scripts/enforce_rulesets.sh) provides:

- **File size checking** - Prevents files >10MB
- **Python syntax validation** - Ensures valid syntax
- **Debug statement detection** - Warns about print statements
- **Trailing whitespace** - Detects unwanted whitespace
- **Gitignore validation** - Ensures proper patterns
- **Branch naming** - Enforces conventional names (feature/*, bugfix/*)
- **Commit message format** - Suggests conventional commits
- **TODO/FIXME tracking** - Lists unresolved comments

## Configuration Files

### `.pre-commit-config.yaml`

Defines all pre-commit hooks that run before commits. Each hook can auto-fix issues or report them.

### `.github/rulesets.json`

Documents all ruleset definitions and enforcement rules in a machine-readable format.

### `scripts/enforce_rulesets.sh`

Bash script that implements additional rules not covered by pre-commit hooks.

### `scripts/manage_rulesets.py`

Python script for managing rulesets programmatically (admin access required for GitHub API).

## Ruleset Rules

| Rule | Tool | Enforcement | Auto-Fix |
| --- | --- | --- | --- |
| Trailing whitespace | pre-commit | ✓ | Yes |
| Large files | pre-commit | ✓ | No |
| YAML validation | pre-commit | ✓ | Yes |
| JSON validation | pre-commit | ✓ | Yes |
| Code formatting | Black | ✓ | Yes |
| Import sorting | isort | ✓ | Yes |
| Code linting | Flake8 | ✓ | No |
| Type checking | MyPy | Manual | No |
| Security scanning | Bandit | ✓ | No |
| Secrets detection | Detect Secrets | ✓ | No |
| Markdown linting | Markdownlint | ✓ | Yes |
| GitHub Actions | Actionlint | ✓ | No |
| File size limit | Custom | ✓ | No |
| Debug statements | Custom | ✓ | No |
| Branch naming | Custom | ✓ | No |
| Commit messages | Custom | Warning | No |

## Typical Workflow

### Before Making Changes

```bash
# Update pre-commit hooks to latest versions
pre-commit autoupdate
```

### While Developing

```bash
# Pre-commit hooks run automatically before each commit
git add .
git commit -m "feat: add new feature"  # Hooks run automatically

# If hooks fail, fix issues and commit again
```

### Before Pushing

```bash
# Run full ruleset check
bash scripts/enforce_rulesets.sh

# Fix any issues and commit
git push origin feature/my-feature
```

### Running Specific Checks

```bash
# Run only Python linting
pre-commit run flake8 --all-files

# Run only Black formatting
pre-commit run black --all-files

# Run only security scanning
pre-commit run bandit --all-files
```

## Troubleshooting

### Pre-commit hooks not running

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install
```

### Bypass hooks temporarily (use sparingly!)

```bash
git commit --no-verify
```

### Update hooks to latest versions

```bash
pre-commit autoupdate
```

### Check hook status

```bash
pre-commit run --all-files --verbose
```

## Comparison with GitHub API Rulesets

| Feature | GitHub API | Open Source Tools |
| --- | --- | --- |
| Cost | Free | Free |
| Setup | Admin access required | No special access |
| Availability | GitHub only | Works anywhere |
| Local enforcement | No | Yes (pre-commit) |
| Auto-fix capability | Limited | Yes (most tools) |
| Customization | Limited | Highly customizable |
| CI/CD integration | Built-in | Via workflows |
| Offline support | No | Yes |

## Contributing

When adding new rules:

1. Add to `.pre-commit-config.yaml` for automated hooks, OR
2. Add function to `scripts/enforce_rulesets.sh` for custom logic
3. Update `.github/rulesets.json` with documentation
4. Test with: `bash scripts/enforce_rulesets.sh`
5. Commit changes

## References

- Pre-commit: <https://pre-commit.com/>
- Black: <https://black.readthedocs.io/>
- Flake8: <https://flake8.pycqa.org/>
- MyPy: <https://mypy.readthedocs.io/>
- Bandit: <https://bandit.readthedocs.io/>
- Detect Secrets: <https://detect-secrets.readthedocs.io/>

## License

All tools are open-source and free to use. See individual tool repositories for specific licenses.
