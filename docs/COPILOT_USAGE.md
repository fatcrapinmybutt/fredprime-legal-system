# GitHub Copilot Usage Guide

## Overview

This repository includes production-grade GitHub Copilot Agent Instructions that enforce engineering excellence and domain-specific requirements for the FRED Supreme Litigation OS.

## Instruction Files

The Copilot instructions are available in two locations:

1. **`.github/copilot-instructions.md`** - Primary location for GitHub Copilot
2. **`COPILOT_AGENT.md`** - Alternative location in repository root

Both files are identical and contain the same comprehensive instruction set.

## What These Instructions Do

The Copilot Agent Instructions configure GitHub Copilot to operate as a **Senior Principal Engineer** with expertise in:

- Production-quality software engineering
- Legal system automation and compliance
- Security-first design principles
- Michigan Court Rules (MCR) compliance
- Evidence chain-of-custody management

## Key Enforcement Areas

### 1. Code Quality Standards

- **No placeholders**: No TODOs, stubs, or incomplete implementations
- **Fail-closed**: All systems must fail safely with explicit error handling
- **Deterministic**: All solutions must be reproducible and testable

### 2. Security & Compliance

- **Legal compliance**: MCR 1.109(D)(3) signature blocks, court deadlines
- **Evidence integrity**: Chain-of-custody preservation, audit logging
- **Data protection**: No secrets in code, validated file paths, sanitized inputs

### 3. Architecture & Design

- **System thinking**: Consider entire repositories, not just individual files
- **Explicit architecture**: Document intent, responsibilities, and data flow
- **Extension over rewrite**: Prefer additive changes and backward compatibility

### 4. Python Best Practices

- **Type hints**: Required for all public APIs
- **PEP 8 compliance**: Enforced via Black, isort, Flake8
- **Error handling**: Explicit, logged, with actionable messages
- **Testing**: pytest with >70% coverage requirement

## How to Use

### For GitHub Copilot Users

GitHub Copilot will automatically detect and use the instructions in `.github/copilot-instructions.md` when:

1. You have GitHub Copilot enabled in your IDE (VS Code, JetBrains, etc.)
2. You're working in this repository
3. The file is in the `.github/` directory

No additional configuration is required!

### For GitHub Copilot Workspace

If using GitHub Copilot Workspace or CLI:

```bash
# The instructions are automatically loaded from:
# - .github/copilot-instructions.md (primary)
# - COPILOT_AGENT.md (fallback)
```

### Manual Reference

You can also reference these instructions manually when prompting Copilot:

```
@workspace Follow the production-grade instructions in .github/copilot-instructions.md
```

## What to Expect

When Copilot generates code with these instructions active:

✅ **Full implementations** - No stubs or TODOs
✅ **Error handling** - Try/except blocks with logging
✅ **Type hints** - Proper Python type annotations
✅ **Documentation** - Docstrings following Google style
✅ **Security** - Input validation and safe file operations
✅ **Testing** - Pytest test cases with appropriate markers
✅ **MCR compliance** - Legal document validation

## Repository-Specific Features

### Legal Domain Awareness

Copilot understands:

- Michigan Court Rules (MCR) requirements
- Court form structure and validation
- Evidence handling protocols
- Chain-of-custody requirements
- FOIA request formats

### Build System Integration

Copilot knows about:

```bash
make test              # Run pytest
make lint              # Run code quality checks
make format            # Auto-format code
make security-check    # Run Bandit security scan
make check             # Run all quality checks
```

### Architecture Understanding

Copilot recognizes:

- Core system components (`core/`, `modules/`, `cli/`)
- Plugin architecture patterns
- Evidence authentication requirements
- Document generation workflows
- Court filing integrations

## Examples

### Code Generation Example

**Prompt:** "Create a function to validate MCR signature blocks"

**Expected Output:**
```python
from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)

def validate_mcr_signature_block(signature_block: str) -> tuple[bool, Optional[str]]:
    """Validate signature block compliance with MCR 1.109(D)(3).
    
    Args:
        signature_block: The signature block text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if compliant, False otherwise
        - error_message: None if valid, description of issue if invalid
        
    Example:
        >>> result = validate_mcr_signature_block(signature_text)
        >>> if not result[0]:
        ...     logger.error(f"Signature validation failed: {result[1]}")
    """
    if not signature_block or not signature_block.strip():
        error = "Signature block cannot be empty"
        logger.error(error)
        return False, error
    
    # MCR 1.109(D)(3) requirements
    required_fields = ["name", "bar number", "address", "phone"]
    
    # Implementation with full validation logic...
    # (Full code would be generated, not truncated)
```

### Documentation Example

**Prompt:** "Document the evidence chain-of-custody workflow"

**Expected Output:** Complete markdown documentation with:
- System overview
- Component responsibilities
- Security requirements
- Failure handling
- Usage examples

### Refactoring Example

**Prompt:** "Refactor this function to improve error handling"

**Expected Output:**
- Preserved external behavior
- Added explicit error handling
- Improved logging
- Type hints added
- Docstring updated
- Test cases suggested

## Troubleshooting

### Copilot Not Following Instructions

If Copilot seems to ignore the instructions:

1. **Verify file location**: Ensure `.github/copilot-instructions.md` exists
2. **Restart IDE**: Reload VS Code or your editor
3. **Check Copilot version**: Update to latest version
4. **Be explicit**: Reference instructions in your prompt

### Instructions Too Strict

The instructions are designed for production code. If you need to:

- **Prototype quickly**: Explicitly tell Copilot "create a prototype with TODOs"
- **Skip tests**: State "skip test generation for this prototype"
- **Simplify**: Request "simplified version for proof-of-concept"

### Custom Overrides

You can override specific requirements by being explicit:

```
Create a simple example (skip error handling for now)
```

## Best Practices

### Effective Prompting

✅ **Good prompts:**
- "Create a production-ready function to parse court dates"
- "Add error handling to this evidence validation logic"
- "Generate pytest tests for the document generator"

❌ **Less effective:**
- "Write some code for dates"
- "Fix this" (without context)
- "Quick hack for validation"

### Working with Instructions

1. **Trust the process**: Instructions ensure production quality
2. **Provide context**: More context = better results
3. **Review output**: Always review generated code
4. **Iterate**: Refine prompts if output isn't quite right

## Updating Instructions

### When to Update

Consider updating instructions when:

- New compliance requirements emerge
- Architecture patterns change
- Security standards evolve
- New tools are adopted

### How to Update

1. Edit `.github/copilot-instructions.md`
2. Sync changes to `COPILOT_AGENT.md`
3. Test with sample prompts
4. Document changes in pull request
5. Update this guide if needed

### Validation

After updates, verify:

```bash
# Ensure files are identical
md5sum .github/copilot-instructions.md COPILOT_AGENT.md

# Check markdown formatting
make format  # if markdown formatting is configured
```

## Additional Resources

- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Main Instructions](.github/copilot-instructions.md)
- [Repository README](../README.md)

## Support

For questions or issues:

1. **Technical questions**: Open a Discussion on GitHub
2. **Instruction bugs**: Open an Issue
3. **Enhancement suggestions**: Start a Discussion with "Copilot:" prefix

---

**Note**: These instructions represent the engineering standards for this repository. They are designed to ensure code quality, security, and legal compliance for production legal system automation.
