# Advanced Development Guide

This guide covers advanced development topics and best practices for the FRED Supreme Litigation OS project.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Module Structure](#module-structure)
- [Testing Strategy](#testing-strategy)
- [Performance Optimization](#performance-optimization)
- [Debugging and Profiling](#debugging-and-profiling)
- [Release Management](#release-management)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

The FRED Supreme Litigation OS follows a modular architecture:

```
fredprime-legal-system/
├── src/                    # Core library code
│   ├── __init__.py
│   └── litigation_cli.py
├── modules/                # Feature modules
│   ├── affidavit_builder.py
│   ├── motion_generator.py
│   ├── binder_exporter.py
│   └── codex_manifest.py
├── cli/                    # Command-line interfaces
├── scripts/                # Utility scripts
├── tests/                  # Test suite
├── docs/                   # Documentation
└── config/                 # Configuration files
```

### Design Principles

1. **Modularity** - Each module has single responsibility
2. **Testability** - Code designed for easy testing
3. **Reusability** - Components can be used independently
4. **Documentation** - Every module documented
5. **Type Safety** - Type hints throughout
6. **Error Handling** - Comprehensive error management

## Module Structure

### Creating a New Module

1. **Create module directory**:
   ```
   modules/my_feature/
   ├── __init__.py
   ├── core.py
   ├── utils.py
   ├── exceptions.py
   └── tests/
       └── test_core.py
   ```

2. **Add module docstring** (core.py):
   ```python
   """My Feature Module
    
   This module provides functionality for...
    
   Classes:
       MyFeature: Main feature class
       
   Functions:
       my_function: Description
       
   Exceptions:
       MyFeatureError: Custom exception
   """
   ```

3. **Implement with type hints**:
   ```python
   from typing import Optional, List, Dict
   from dataclasses import dataclass
   
   @dataclass
   class FeatureConfig:
       """Configuration for feature."""
       name: str
       enabled: bool = True
   
   class MyFeature:
       """Main feature class."""
       
       def __init__(self, config: FeatureConfig) -> None:
           """Initialize feature."""
           self.config = config
       
       def process(self, data: Dict[str, Any]) -> Optional[str]:
           """Process data."""
           pass
   ```

4. **Add comprehensive tests**:
   ```python
   import pytest
   from modules.my_feature import MyFeature, FeatureConfig
   
   class TestMyFeature:
       @pytest.fixture
       def config(self):
           return FeatureConfig(name="test")
       
       def test_initialization(self, config):
           feature = MyFeature(config)
           assert feature.config == config
   ```

5. **Update __init__.py**:
   ```python
   """My Feature Module"""
   from .core import MyFeature, FeatureConfig
   
   __all__ = ["MyFeature", "FeatureConfig"]
   ```

## Testing Strategy

### Test Pyramid

```
        Unit Tests (70%)
      /                  \
    Integration Tests (20%)
    /                      \
   E2E Tests (10%)
```

### Test Organization

```
tests/
├── unit/                      # Unit tests
│   ├── test_module_a.py
│   └── test_module_b.py
├── integration/               # Integration tests
│   └── test_workflow.py
├── e2e/                       # End-to-end tests
│   └── test_full_process.py
└── fixtures/                  # Test data
    └── sample_documents/
```

### Writing Tests

#### Unit Tests (Fast, Isolated)

```python
import pytest
from modules.core import process_document

def test_process_valid_document():
    """Test processing valid document."""
    result = process_document(valid_doc)
    assert result is not None
    assert result['status'] == 'success'

def test_process_invalid_document():
    """Test processing invalid document."""
    with pytest.raises(ValueError):
        process_document(invalid_doc)
```

#### Integration Tests (Medium Speed)

```python
@pytest.mark.integration
def test_document_pipeline():
    """Test full document processing pipeline."""
    doc = create_test_document()
    manager = DocumentManager()
    result = manager.process_and_validate(doc)
    assert result.is_valid()
```

#### E2E Tests (Slower)

```python
@pytest.mark.slow
def test_end_to_end_litigation_workflow():
    """Test complete litigation workflow."""
    # Create case
    case = create_case("vs Smith")
    # Add documents
    case.add_document(complaint)
    # Generate forms
    forms = case.generate_court_forms()
    assert len(forms) > 0
```

### Test Coverage

```bash
# Generate coverage report
make test-coverage

# View HTML report
open htmlcov/index.html

# Set minimum coverage threshold in pyproject.toml
```

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

def profile_function():
    """Profile a function."""
    pr = cProfile.Profile()
    pr.enable()
    
    # Code to profile
    process_large_document()
    
    pr.disable()
    ps = pstats.Stats(pr)
    ps.sort_stats('cumulative')
    ps.print_stats(10)
```

### Benchmarking

```python
import timeit

# Simple timing
start = time.time()
result = expensive_operation()
elapsed = time.time() - start
print(f"Elapsed: {elapsed:.4f}s")

# Comparative benchmarking
time1 = timeit.timeit(approach1, number=1000)
time2 = timeit.timeit(approach2, number=1000)
print(f"Improvement: {(time1/time2 - 1) * 100:.1f}%")
```

### Optimization Tips

1. **Use generators for large datasets**:
   ```python
   # Bad: Loads all in memory
   documents = [load_document(f) for f in files]
   
   # Good: Lazy loading
   def load_documents(files):
       for f in files:
           yield load_document(f)
   ```

2. **Cache expensive operations**:
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def expensive_calculation(param):
       return result
   ```

3. **Use type hints for better performance**:
   ```python
   # Type hints help Python optimize
   def process(data: List[Dict[str, str]]) -> int:
       pass
   ```

## Debugging and Profiling

### Debug Mode

```python
import logging

logger = logging.getLogger(__name__)
logger.debug("Debug message")

# Enable debug
logging.basicConfig(level=logging.DEBUG)
```

### Using pdb (Python Debugger)

```python
import pdb

def complex_function(data):
    pdb.set_trace()  # Breakpoint here
    result = do_something(data)
    return result
```

### Common Debugging Commands

```
(Pdb) n         # Next line
(Pdb) s         # Step into
(Pdb) c         # Continue
(Pdb) l         # List code
(Pdb) p var     # Print variable
(Pdb) pp dict   # Pretty print
(Pdb) h         # Help
```

## Release Management

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- MAJOR.MINOR.PATCH (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Process

1. **Update version**:
   ```bash
   # In pyproject.toml and setup.cfg
   version = "1.2.3"
   ```

2. **Update CHANGELOG.md**:
   ```markdown
   ## [1.2.3] - 2026-01-20
   ### Added
   - New features
   ### Fixed
   - Bug fixes
   ```

3. **Commit and tag**:
   ```bash
   git add .
   git commit -m "chore: release v1.2.3"
   git tag -a v1.2.3 -m "Release v1.2.3"
   git push origin main
   git push origin v1.2.3
   ```

4. **Build distribution**:
   ```bash
   make build
   ```

5. **Upload (if publishing)**:
   ```bash
   make upload  # Requires PyPI credentials
   ```

## Troubleshooting

### Common Issues

#### Tests Failing Locally but Passing in CI

- **Check Python version**: `python --version`
- **Rebuild dependencies**: `pip install -e ".[dev]" --force-reinstall`
- **Clear cache**: `make clean-cache`
- **Run pre-commit**: `pre-commit run --all-files`

#### Import Errors

- **Check module path**: Ensure module in PYTHONPATH
- **Check imports**: Verify all imports exist
- **Reload**: `python -c "import sys; sys.path.insert(0, '.'); import module"`

#### Type Checking Errors

- **Run mypy directly**: `mypy src modules --show-error-codes`
- **Add type ignores if needed**: `# type: ignore`
- **Update stubs**: `pip install types-<package>`

#### Performance Issues

- **Profile code**: Use cProfile
- **Check for N+1 queries**: Review database calls
- **Monitor memory**: Use memory_profiler
- **Optimize hot paths**: Focus on most-used code

### Getting Help

1. **Check existing issues**: Search GitHub Issues
2. **Enable debug logging**: Set `DEBUG=True`
3. **Run diagnostics**: `python -m pip check`
4. **Create detailed bug report**: Include environment and steps

### Useful Commands

```bash
# Check project structure
tree -I '__pycache__|*.pyc|.git|venv' -L 2

# Find Python syntax errors
python -m py_compile src/**/*.py

# Check imports
python -c "import sys; import modules"

# Run linters with verbose output
flake8 --statistics --count

# Generate requirements from imports
pipreqs . --force

# Check for security issues
bandit -r . -f json > security-report.json
```

---

For more information, see:
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/OPEN_SOURCE_RULESETS.md](docs/OPEN_SOURCE_RULESETS.md)
- Project README for quick start
