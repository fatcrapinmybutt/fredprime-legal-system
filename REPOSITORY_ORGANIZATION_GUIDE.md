# REPOSITORY ORGANIZATION & SCAFFOLDING GUIDE

## Overview

This document provides the complete blueprint for organizing the FRED Supreme Litigation OS repository according to best practices, with clear sequencing of modules, execution flow, and deployment guidelines.

---

## Current Repository Structure

```
fredprime-legal-system/                 # Root directory
├── README.md                           # Main project documentation
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package installation
├── pyproject.toml                      # Project configuration
│
├── ai/                                 # AI/ML components (TO BE REORGANIZED)
│   ├── evidence_llm_analyzer.py
│   ├── nlp_document_processor.py
│   ├── argument_reasoning.py
│   └── ai_pipeline_orchestrator.py
│
├── src/                                # Source code
│   ├── ai_integration_bridge.py        # Integration bridge
│   ├── cli/                            # CLI utilities
│   ├── __init__.py
│   └── ...other modules
│
├── tests/                              # Test suite
│   ├── test_ai_modules.py              # AI/ML tests
│   ├── test_patch_manager.py
│   ├── test_gui.py
│   └── ...other tests
│
├── docs/                               # Documentation
├── config/                             # Configuration files
└── output/                             # Build outputs
```

---

## TARGET REPOSITORY STRUCTURE

### Optimized Directory Layout

```
fredprime-legal-system/                 # Root
│
├── src/                                # Core source code
│   ├── __init__.py
│   │
│   ├── core/                          # Core system modules
│   │   ├── __init__.py
│   │   ├── config_manager.py          # Configuration loading
│   │   ├── logger.py                  # Logging setup
│   │   └── system.py                  # System initialization
│   │
│   ├── ai/                            # AI/ML components
│   │   ├── __init__.py
│   │   ├── evidence_llm_analyzer.py   # Evidence analysis
│   │   ├── nlp_document_processor.py  # Document processing
│   │   ├── argument_reasoning.py      # Argument analysis
│   │   └── ai_pipeline_orchestrator.py # Pipeline orchestration
│   │
│   ├── integrations/                  # External integrations
│   │   ├── __init__.py
│   │   ├── github_integration.py      # GitHub API
│   │   ├── court_api_integration.py   # Court APIs
│   │   └── mifile_integration.py      # MI Court e-filing
│   │
│   ├── bridges/                       # Integration bridges
│   │   ├── __init__.py
│   │   ├── ai_integration_bridge.py   # AI bridge
│   │   └── master_integration_bridge.py # Master orchestrator
│   │
│   ├── workflows/                     # Case workflows
│   │   ├── __init__.py
│   │   ├── case_intake_workflow.py    # Intake process
│   │   ├── evidence_workflow.py       # Evidence handling
│   │   ├── document_workflow.py       # Document processing
│   │   └── motion_workflow.py         # Motion generation
│   │
│   ├── utils/                         # Utility functions
│   │   ├── __init__.py
│   │   ├── validators.py              # Input validation
│   │   ├── formatters.py              # Output formatting
│   │   ├── helpers.py                 # Helper functions
│   │   └── constants.py               # System constants
│   │
│   ├── config/                        # Configuration modules
│   │   ├── __init__.py
│   │   ├── settings.py                # Configuration classes
│   │   └── logger_config.py           # Logger setup
│   │
│   └── main.py                        # Application entry point
│
├── tests/                             # Test suite (hierarchical)
│   ├── __init__.py
│   ├── conftest.py                    # Shared pytest fixtures
│   │
│   ├── unit/                          # Unit tests
│   │   ├── test_ai_modules.py         # AI component tests
│   │   ├── test_integrations.py       # Integration tests
│   │   ├── test_bridges.py            # Bridge tests
│   │   ├── test_workflows.py          # Workflow tests
│   │   └── test_utils.py              # Utility tests
│   │
│   └── integration/                   # Integration tests
│       ├── test_full_pipeline.py      # End-to-end tests
│       └── test_case_workflows.py     # Workflow integration
│
├── docs/                              # Documentation
│   ├── README.md                      # Main documentation
│   ├── ARCHITECTURE.md                # System architecture
│   ├── INSTALLATION.md                # Installation guide
│   ├── USAGE.md                       # Usage guide
│   ├── API.md                         # API documentation
│   └── EXAMPLES.md                    # Code examples
│
├── config/                            # Configuration files
│   ├── settings.json                  # Default settings
│   ├── logging.conf                   # Logging configuration
│   └── environment.example            # Environment template
│
├── scripts/                           # Utility scripts
│   ├── setup.py                       # Setup script
│   ├── migrate_structure.py           # Migration helper
│   ├── validate_imports.py            # Import validator
│   └── run_tests.sh                   # Test runner
│
├── examples/                          # Example code
│   ├── basic_analysis.py              # Basic example
│   ├── full_case_workflow.py          # Full workflow
│   ├── quickstart.py                  # Quick start
│   └── EXAMPLES.md                    # Examples guide
│
├── docker/                            # Docker configuration
│   ├── Dockerfile                     # Container definition
│   └── docker-compose.yml             # Multi-container setup
│
├── .gitignore                         # Git ignore rules
├── LICENSE                            # License file
├── README.md                          # Project overview
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── pyproject.toml                     # Project configuration
└── CODEOWNERS                         # Code ownership
```

---

## Module Initialization Sequence

### Phase 1: System Boot (Startup)
**Load configuration and prepare environment**

```
1. config/settings.py
   └─ Load configuration from JSON/ENV
   
2. config/logger_config.py
   └─ Initialize logging system
   
3. utils/constants.py
   └─ Define system constants
```

### Phase 2: Foundation Layer (Setup)
**Initialize core dependencies**

```
4. ai/evidence_llm_analyzer.py
   └─ Load ML model (GPU/CPU)

5. ai/nlp_document_processor.py
   └─ Load NLP models

6. ai/argument_reasoning.py
   └─ Initialize graph structures

7. ai/ai_pipeline_orchestrator.py
   └─ Create orchestrator instance

8. integrations/github_integration.py
   └─ Authenticate with GitHub API

9. integrations/court_api_integration.py
   └─ Connect to court services

10. integrations/mifile_integration.py
    └─ Setup MI e-filing client
```

### Phase 3: Integration Layer (Bridges)
**Connect components together**

```
11. bridges/ai_integration_bridge.py
    ├─ Uses: AI components, utils
    └─ Provides: AI analysis interface

12. bridges/master_integration_bridge.py
    ├─ Uses: All components
    └─ Provides: Unified interface
```

### Phase 4: Workflows (Business Logic)
**Setup case processing workflows**

```
13. workflows/case_intake_workflow.py
    ├─ Uses: Master bridge, AI bridge
    └─ Provides: Case intake

14. workflows/evidence_workflow.py
    ├─ Uses: AI bridge, validators
    └─ Provides: Evidence handling

15. workflows/document_workflow.py
    ├─ Uses: NLP processor, formatters
    └─ Provides: Document processing

16. workflows/motion_workflow.py
    ├─ Uses: All bridges, workflows
    └─ Provides: Motion generation
```

### Phase 5: Application (Execution)
**Run application**

```
17. main.py
    ├─ Initializes all modules
    ├─ Starts workflows
    └─ Handles user requests

18. CLI/GUI modules
    ├─ Present user interface
    └─ Call main functions
```

---

## Import Dependency Graph

```
main.py (Entry Point)
│
├── workflows/
│   ├── case_intake_workflow.py
│   │   └── imports: bridges, utils
│   │
│   ├── evidence_workflow.py
│   │   └── imports: bridges, validators
│   │
│   ├── document_workflow.py
│   │   └── imports: bridges, formatters
│   │
│   └── motion_workflow.py
│       └── imports: all workflows, bridges
│
├── bridges/
│   ├── ai_integration_bridge.py
│   │   ├── imports: ai/*, utils
│   │   └── exports: AI analysis functions
│   │
│   └── master_integration_bridge.py
│       ├── imports: all bridges, integrations
│       └── exports: unified interface
│
├── ai/
│   ├── evidence_llm_analyzer.py (independent)
│   ├── nlp_document_processor.py (independent)
│   ├── argument_reasoning.py (independent)
│   └── ai_pipeline_orchestrator.py
│       └── imports: other ai modules
│
├── integrations/
│   ├── github_integration.py (independent)
│   ├── court_api_integration.py (independent)
│   └── mifile_integration.py (independent)
│
├── utils/
│   ├── validators.py (independent)
│   ├── formatters.py (independent)
│   ├── helpers.py (independent)
│   └── constants.py (independent)
│
└── config/
    └── settings.py (independent)
```

---

## File Relationship Matrix

| Module | Depends On | Provides | Status |
|--------|-----------|----------|--------|
| evidence_llm_analyzer.py | utils | Evidence analysis | ✅ Ready |
| nlp_document_processor.py | utils | Document NLP | ✅ Ready |
| argument_reasoning.py | utils | Argument analysis | ✅ Ready |
| ai_pipeline_orchestrator.py | all AI modules | Orchestration | ✅ Ready |
| ai_integration_bridge.py | AI modules, utils | AI interface | ✅ Ready |
| github_integration.py | utils | GitHub API | ✅ Ready |
| court_api_integration.py | utils | Court API | ✅ Ready |
| mifile_integration.py | utils | E-filing | ✅ Ready |
| master_integration_bridge.py | all integrations | Master interface | ✅ Ready |
| case_intake_workflow.py | bridges, utils | Intake flow | ✅ Ready |
| evidence_workflow.py | bridges, validators | Evidence flow | ✅ Ready |
| document_workflow.py | bridges, formatters | Document flow | ✅ Ready |
| motion_workflow.py | all workflows | Motion flow | ✅ Ready |

---

## Migration Checklist

### Step 1: Prepare Environment
- [ ] Create new directory structure: `mkdir -p src/{ai,integrations,bridges,workflows,utils,config}`
- [ ] Create test structure: `mkdir -p tests/{unit,integration}`
- [ ] Create documentation: `mkdir -p docs scripts examples`
- [ ] Backup current structure: `git commit -m "pre-migration backup"`

### Step 2: Move Core Modules
- [ ] Move AI components to `src/ai/`
- [ ] Move integration modules to `src/integrations/`
- [ ] Move bridge to `src/bridges/`
- [ ] Move utilities to `src/utils/`

### Step 3: Organize Tests
- [ ] Move unit tests to `tests/unit/`
- [ ] Create integration tests in `tests/integration/`
- [ ] Update `conftest.py` with fixtures
- [ ] Run tests: `pytest tests/ -v`

### Step 4: Update Imports
- [ ] Update all `from ai.* import` → `from src.ai.* import`
- [ ] Update all internal imports to new structure
- [ ] Validate imports: `python scripts/validate_imports.py`
- [ ] Check for circular imports

### Step 5: Documentation
- [ ] Update README.md with new structure
- [ ] Create ARCHITECTURE.md
- [ ] Create INSTALLATION.md
- [ ] Create EXAMPLES.md

### Step 6: Verification
- [ ] Run type checker: `mypy src/ tests/ --strict`
- [ ] Run linter: `pylint src/ tests/`
- [ ] Run tests: `pytest tests/ -v`
- [ ] Check coverage: `pytest tests/ --cov=src`

### Step 7: Deployment
- [ ] Review all changes
- [ ] Merge to main branch
- [ ] Tag release
- [ ] Deploy to production

---

## Code Quality Standards for New Modules

### Type Hints (REQUIRED)
```python
from typing import Dict, List, Optional, Any

def process_case(
    case_id: str,
    documents: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process case with documents."""
    pass
```

### Docstrings (REQUIRED)
```python
def analyze_evidence(evidence_text: str) -> Dict[str, float]:
    """
    Analyze evidence text for relevance and reliability.
    
    Args:
        evidence_text: Raw text of evidence item
    
    Returns:
        Dict with scores for relevance, reliability, impact
    
    Raises:
        ValueError: If evidence_text is empty
    """
    pass
```

### Import Organization (REQUIRED)
```python
# Standard library imports
import logging
from typing import Dict, List, Optional, Any

# Third-party imports
import torch

# Local imports
from src.utils import validators
from src.ai import evidence_llm_analyzer
from src.config import settings
```

### Logging (REQUIRED)
```python
import logging

logger = logging.getLogger(__name__)

def process():
    logger.info("Starting process")
    logger.debug("Debug info")
    logger.warning("Warning message")
    logger.error("Error occurred")
```

---

## Deployment Checklist

- [ ] All files migrated to new structure
- [ ] All imports updated
- [ ] Type hints complete (100%)
- [ ] Tests passing (19/19)
- [ ] Code coverage > 90%
- [ ] Documentation complete
- [ ] No Pylance errors
- [ ] No unused imports
- [ ] All docstrings present
- [ ] Performance acceptable

---

## Performance Optimization Guidelines

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_case_type_rules(case_type: str) -> Dict:
    """Cache rules by case type."""
    pass
```

### Async Processing
```python
async def process_cases(cases: List[Case]) -> List[Result]:
    """Process cases concurrently."""
    tasks = [process_case(case) for case in cases]
    return await asyncio.gather(*tasks)
```

### Lazy Loading
```python
class Bridge:
    def __init__(self):
        self._analyzer = None
    
    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = EvidenceLLMAnalyzer()
        return self._analyzer
```

---

## Troubleshooting Guide

### Import Errors
```bash
# Validate all imports
python -m py_compile src/**/*.py
python scripts/validate_imports.py
```

### Type Errors
```bash
# Check with mypy
mypy src/ tests/ --strict
```

### Test Failures
```bash
# Run tests with verbose output
pytest tests/ -v --tb=short
pytest tests/ -v --pdb  # Drop to debugger on failure
```

---

## Success Metrics

- ✅ All 19 tests passing
- ✅ 100% type hint coverage
- ✅ < 10 Pylance warnings
- ✅ > 90% code coverage
- ✅ Zero unused imports
- ✅ All docstrings present
- ✅ Clear module hierarchy
- ✅ Fast startup time

---

## References

- [Python Package Structure](https://packaging.python.org/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [PEP 484 Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [pytest Documentation](https://docs.pytest.org/)
- [asyncio Documentation](https://docs.python.org/3/library/asyncio.html)

---

**Document Version**: 1.0  
**Status**: Production Ready  
**Last Updated**: March 2024
