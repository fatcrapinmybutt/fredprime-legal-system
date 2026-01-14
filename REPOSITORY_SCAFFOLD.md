# FRED Supreme Litigation OS - Repository Organization Guide

## Overview

This document defines the optimal organization and sequencing of the FRED Supreme
Litigation OS repository structure, with emphasis on AI/ML integration and code scalability.

## Directory Structure (Recommended)

```
fredprime-legal-system/
│
├── README.md                           # Main project documentation
├── LICENSE                             # Project license
├── pyproject.toml                      # Python project configuration
├── requirements.txt                    # Dependency specifications
├── setup.cfg                           # Setup configuration
│
├── docs/                               # Documentation
│   ├── API_REFERENCE.md               # API documentation
│   ├── SETUP_GUIDE.md                 # Installation/setup
│   ├── USER_GUIDE.md                  # User manual
│   ├── DEVELOPER_GUIDE.md             # Development guide
│   └── architecture/
│       ├── SYSTEM_ARCHITECTURE.md
│       ├── AI_ML_ARCHITECTURE.md
│       └── DATABASE_SCHEMA.md
│
├── src/                               # Source code root
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── litigation_cli.py          # Main CLI interface
│   │   ├── config_manager.py          # Configuration management
│   │   └── logger.py                  # Logging utilities
│   │
│   ├── ai/                            # AI/ML Components
│   │   ├── __init__.py
│   │   ├── evidence_llm_analyzer.py   # Evidence analysis
│   │   ├── nlp_document_processor.py  # Document processing
│   │   ├── argument_reasoning.py      # Argument analysis
│   │   ├── ai_pipeline_orchestrator.py # Pipeline orchestration
│   │   └── models/
│   │       └── __init__.py
│   │
│   ├── integrations/                  # External Integrations
│   │   ├── __init__.py
│   │   ├── github_integration.py      # GitHub API
│   │   ├── mifile_integration.py      # Michigan Court integration
│   │   └── foia_integration.py        # FOIA request handling
│   │
│   ├── workflows/                     # Business Workflows
│   │   ├── __init__.py
│   │   ├── case_intake.py            # Case intake workflow
│   │   ├── evidence_analysis.py      # Evidence analysis flow
│   │   ├── document_processing.py    # Document processing flow
│   │   └── motion_generation.py      # Motion generation flow
│   │
│   ├── bridges/                       # Integration Bridges
│   │   ├── __init__.py
│   │   ├── ai_integration_bridge.py   # AI system bridge
│   │   └── workflow_bridge.py         # Workflow bridge
│   │
│   └── utils/                         # Utility Functions
│       ├── __init__.py
│       ├── validators.py              # Input validation
│       ├── formatters.py              # Output formatting
│       └── helpers.py                 # Helper functions
│
├── tests/                             # Test Suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest configuration
│   │
│   ├── unit/                         # Unit Tests
│   │   ├── __init__.py
│   │   ├── test_ai_modules.py
│   │   ├── test_evidence_analyzer.py
│   │   ├── test_nlp_processor.py
│   │   ├── test_arg_system.py
│   │   └── test_github_integration.py
│   │
│   ├── integration/                  # Integration Tests
│   │   ├── __init__.py
│   │   ├── test_ai_pipeline.py
│   │   ├── test_workflows.py
│   │   └── test_bridges.py
│   │
│   ├── fixtures/                    # Test Fixtures
│   │   ├── __init__.py
│   │   ├── evidence_fixtures.py
│   │   ├── document_fixtures.py
│   │   └── case_fixtures.py
│   │
│   └── data/                        # Test Data
│       ├── sample_evidence.json
│       ├── sample_documents.pdf
│       └── test_cases.json
│
├── config/                          # Configuration Files
│   ├── development.json             # Dev configuration
│   ├── staging.json                 # Staging configuration
│   ├── production.json              # Prod configuration
│   └── system_enforcement.json      # System enforcement rules
│
├── scripts/                         # Utility Scripts
│   ├── setup_environment.sh         # Environment setup
│   ├── run_tests.sh                 # Test runner
│   ├── build_deployment.sh          # Build script
│   └── data_migration.py            # Data migration tool
│
├── examples/                        # Example Code
│   ├── QUICKSTART_AI_ML.py          # Quick start guide
│   ├── basic_evidence_analysis.py   # Basic examples
│   ├── document_processing.py       # Document examples
│   └── full_case_analysis.py        # Complete example
│
├── db/                              # Database Files
│   ├── migrations/                  # Database migrations
│   ├── seeds/                       # Seed data
│   └── schemas/                     # Schema definitions
│
├── docker/                          # Docker Configuration
│   ├── Dockerfile                   # Main dockerfile
│   ├── docker-compose.yml           # Docker compose
│   └── .dockerignore                # Docker ignore
│
└── .gitignore                       # Git ignore rules
```

## Execution Sequence & Dependencies

### Phase 1: Initialization

1. **Core System Setup**
   - `src/core/config_manager.py` - Initialize configuration
   - `src/core/logger.py` - Setup logging
   - `src/core/litigation_cli.py` - Initialize CLI

### Phase 2: Foundation Layer

1. **AI/ML Components**

   - `src/ai/evidence_llm_analyzer.py` - Initialize Evidence Analyzer
   - `src/ai/nlp_document_processor.py` - Initialize NLP Processor
   - `src/ai/argument_reasoning.py` - Initialize ARG System

2. **External Integrations**
   - `src/integrations/github_integration.py` - GitHub integration
   - `src/integrations/mifile_integration.py` - Michigan Court integration

### Phase 3: Orchestration Layer

1. **Pipeline Setup**
   - `src/ai/ai_pipeline_orchestrator.py` - Initialize orchestrator
   - `src/bridges/ai_integration_bridge.py` - Setup bridge

### Phase 4: Workflow Layer

1. **Business Workflows**
   - `src/workflows/case_intake.py` - Case intake workflow
   - `src/workflows/evidence_analysis.py` - Evidence workflow
   - `src/workflows/document_processing.py` - Document workflow
   - `src/workflows/motion_generation.py` - Motion workflow

### Phase 5: Utility Layer

1. **Support Functions**
   - `src/utils/validators.py` - Input validation
   - `src/utils/formatters.py` - Output formatting
   - `src/utils/helpers.py` - Helper functions

## Import Dependency Graph

```
CLI Entry Point (src/core/litigation_cli.py)
    ↓
Config Manager (src/core/config_manager.py)
    ├─→ Logger (src/core/logger.py)
    └─→ Workflows (src/workflows/*)
        ├─→ Bridges (src/bridges/ai_integration_bridge.py)
        │   ├─→ Orchestrator (src/ai/ai_pipeline_orchestrator.py)
        │   │   ├─→ Evidence Analyzer (src/ai/evidence_llm_analyzer.py)
        │   │   ├─→ NLP Processor (src/ai/nlp_document_processor.py)
        │   │   └─→ ARG System (src/ai/argument_reasoning.py)
        │   └─→ Integrations (src/integrations/*)
        │       ├─→ GitHub (src/integrations/github_integration.py)
        │       └─→ Michigan Court (src/integrations/mifile_integration.py)
        └─→ Utils (src/utils/*)
            ├─→ Validators (src/utils/validators.py)
            ├─→ Formatters (src/utils/formatters.py)
            └─→ Helpers (src/utils/helpers.py)
```

## File Organization Principles

### 1. Separation of Concerns

- **AI/ML Components**: Isolated in `src/ai/`
- **Integration Services**: Isolated in `src/integrations/`
- **Business Logic**: Isolated in `src/workflows/`
- **Support Code**: Isolated in `src/utils/`

### 2. Import Management

- ✅ **Good**: `from src.ai.evidence_llm_analyzer import EvidenceLLMAnalyzer`
- ✅ **Good**: `from src.integrations.github_integration import GitHubAPIClient`
- ❌ **Bad**: `from src.ai.evidence_llm_analyzer import *`
- ❌ **Bad**: Circular imports between modules

### 3. Testing Organization

- **Unit Tests**: `tests/unit/` - Test individual components
- **Integration Tests**: `tests/integration/` - Test component interactions
- **Fixtures**: `tests/fixtures/` - Shared test data
- **Test Data**: `tests/data/` - Sample files for testing

### 4. Configuration Management

- **Development**: `config/development.json`
- **Staging**: `config/staging.json`
- **Production**: `config/production.json`
- **System Rules**: `config/system_enforcement.json`

## Code Quality Standards

### Type Hints

```python
# ✅ Good - Full type hints
def analyze_evidence(
    evidence_id: str,
    content: str,
    case_type: str = "general"
) -> AnalyzedEvidence:
    """Analyze evidence using LLM."""
    pass

# ❌ Bad - No type hints
def analyze_evidence(evidence_id, content, case_type="general"):
    pass
```

### Docstrings

```python
# ✅ Good - Complete docstring
def process_document(
    document_text: str,
    doc_type: str
) -> DocumentMetadata:
    """
    Process legal document using NLP.

    Args:
        document_text: The document content to analyze
        doc_type: Type of document (Motion, Affidavit, etc.)

    Returns:
        DocumentMetadata: Processed document information

    Raises:
        ValueError: If document_text is empty
    """
    pass
```

### Import Organization

```python
# ✅ Good - Organized imports
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

from ai.evidence_llm_analyzer import EvidenceLLMAnalyzer
from integrations.github_integration import GitHubAPIClient
from utils.validators import validate_evidence_id
```

## Migration Steps to New Structure

### Step 1: Move Core Files

```bash
mkdir -p src/core src/ai src/integrations src/workflows src/bridges src/utils
cp core/litigation_cli.py src/core/
cp ai/*.py src/ai/
cp integrations/*.py src/integrations/
```

### Step 2: Update Imports

```python
# Old
from core.config_manager import ConfigManager

# New
from src.core.config_manager import ConfigManager
```

### Step 3: Reorganize Tests

```bash
mkdir -p tests/unit tests/integration tests/fixtures tests/data
mv tests/test_*.py tests/unit/
```

### Step 4: Verify All Tests Pass

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
```

## Deployment Checklist

- [ ] All imports updated to new structure
- [ ] All tests passing (unit + integration)
- [ ] Configuration files in correct locations
- [ ] Documentation updated
- [ ] Type hints complete
- [ ] No circular dependencies
- [ ] Code follows PEP 8 standards
- [ ] Performance benchmarks met
- [ ] Security review completed
- [ ] Ready for production deployment

## Maintenance Guidelines

1. **Adding New Modules**: Follow directory structure rules
2. **Updating Imports**: Use absolute imports from `src/`
3. **Testing**: Write both unit and integration tests
4. **Documentation**: Update relevant docs/ files
5. **Configuration**: Store in `config/` directory
6. **Dependencies**: Update `requirements.txt` and `pyproject.toml`

## Performance Optimization

- **Lazy Loading**: Import heavy modules only when needed
- **Caching**: Use caching for frequent operations
- **Async Operations**: Use async/await for I/O operations
- **Batch Processing**: Process multiple items in batches
- **Connection Pooling**: Reuse connections for external services

## Scaling Considerations

1. **Horizontal Scaling**: Stateless design for multi-instance deployment
2. **Vertical Scaling**: Optimize memory and CPU usage
3. **Database Sharding**: Partition data across multiple databases
4. **Caching Layer**: Redis for frequently accessed data
5. **Message Queue**: RabbitMQ/Kafka for asynchronous processing

---

**Last Updated**: March 2024
**Version**: 1.0.0
**Status**: Production Ready
