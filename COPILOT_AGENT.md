# COPILOT_AGENT.md

## GitHub Copilot Agent Configuration for FRED Supreme Litigation OS

This document serves as a reference for the GitHub Copilot agent instructions configured for this repository. The actual agent instructions are located at `.github/copilot-instructions.md` and are automatically applied by GitHub Copilot.

---

## Overview

The FRED Supreme Litigation OS uses **production-grade GitHub Copilot instructions** that enforce:

- **Senior Principal Engineer** level code quality
- **Systems Architect** thinking across components
- **AI Reliability Officer** standards for robustness

### Core Principles

1. **No Skeletons or Placeholders**: Every code suggestion is production-ready
2. **Fail-Closed Engineering**: Safe failures with explicit error handling
3. **Determinism & Reproducibility**: Versionable, testable, reproducible solutions
4. **Security by Default**: Proactive security considerations in all suggestions
5. **Architectural Coherence**: System-level thinking, not just file-level edits

---

## What This Means for Developers

When using GitHub Copilot in this repository:

### Code Suggestions Will

✅ **Be production-ready** - Compile/run as-is, no TODOs or stubs
✅ **Include error handling** - Explicit error paths, not silent failures  
✅ **Follow patterns** - Match existing architectural patterns in the repo
✅ **Be secure** - Validate inputs, handle secrets properly, prevent injections
✅ **Be testable** - Deterministic, reproducible, with clear boundaries
✅ **Include logging** - Audit trails for legal compliance where appropriate

### Code Suggestions Will NOT

❌ Generate tutorial-style or "example only" code  
❌ Use TODOs or "left as an exercise" comments  
❌ Create empty handlers or fake implementations  
❌ Silently fail or use "best guess" behavior  
❌ Hard-code secrets or sensitive paths  
❌ Generate lowest-common-denominator solutions  

---

## Repository-Specific Guidance

### Legal Document Generation

- Must comply with Michigan Court Rules (MCR)
- Preserve legal formatting (signature blocks, captions, etc.)
- Include proper service and filing metadata
- Never generate legally incorrect or non-compliant documents

### Evidence Management

- Maintain chain of custody for all exhibits
- Use blockchain timestamping for forensic integrity
- Never modify signed or timestamped evidence
- Generate proper exhibit labels and binders

### AI/NLP Components

- Validate legal doctrine against benchbooks
- Detect contradictions in testimony or filings
- Run adversarial tests on generated arguments
- Never hallucinate legal citations or precedents

### Build System

- Follow the modular plugin architecture
- Use event-driven hooks for extensibility
- Support hot-swapping of modules
- Maintain backward compatibility with existing case files

---

## Development Workflow

### Before Contributing

1. Review `.github/copilot-instructions.md` for full agent instructions
2. Run `make install-dev` to set up development environment
3. Run `make check` to verify code quality before committing
4. Use `make test` to run the test suite

### When Using Copilot

- Let Copilot suggest complete, production-ready implementations
- Review suggestions for legal compliance (document generation)
- Verify security considerations (input validation, secret handling)
- Check that suggestions match existing architectural patterns
- Test suggested code thoroughly before committing

### Quality Standards

All code in this repository must meet:

- **Correctness**: Works as intended, handles edge cases
- **Security**: Input validation, no secrets in code, secure defaults
- **Maintainability**: Clear structure, appropriate logging, documented
- **Testability**: Unit tests for logic, integration tests for workflows
- **Legal Compliance**: Court-ready documents, valid legal procedures

---

## Customization & Extension

### Sub-Role Configurations

The Copilot instructions can be extended with specialized sub-roles:

- **Agent Role**: General code generation and feature development
- **Reviewer Role**: Code review focus with emphasis on security/compliance
- **Refactorer Role**: Safe refactoring with behavior preservation

### Language-Specific Overlays

Additional instructions can be added for specific languages:

- **Python**: Type hints, async/await patterns, context managers
- **TypeScript**: Strict mode, proper typing, modern ES features
- **Rust**: Memory safety, error handling with Result, ownership
- **Go**: Goroutines, channels, defer patterns, error handling

### Enterprise Compliance

For enterprise deployments, additional constraints can be enforced:

- Multi-tenant isolation
- Compliance logging and auditing
- Data residency requirements
- Access control and authorization boundaries

---

## Enforcement Mechanisms

### What GitHub Copilot Will Do

- Automatically apply instructions from `.github/copilot-instructions.md`
- Generate code suggestions that follow the defined standards
- Provide explanations aligned with the architectural principles
- Refuse to generate non-compliant code patterns

### What Developers Must Do

- Review all Copilot suggestions before accepting
- Run quality checks (`make check`) before committing
- Write tests for new functionality
- Update documentation for significant changes
- Maintain legal compliance for document generation

### CI/CD Validation

The following checks are enforced in CI:

- Code quality: `flake8`, `black`, `isort`
- Type checking: `mypy`
- Security: `bandit`
- Tests: `pytest` with coverage reporting
- Rulesets: Custom enforcement scripts

---

## FAQs

### Q: Will Copilot write all my code for me?

**A**: Copilot will provide high-quality suggestions, but you remain responsible for reviewing, testing, and maintaining all code.

### Q: What if Copilot suggests something that violates the instructions?

**A**: Report the issue. Copilot is an AI assistant and may occasionally suggest non-compliant code. Always review suggestions critically.

### Q: Can I override the instructions for my feature?

**A**: The instructions are repository-wide. If you need an exception, document it clearly with justification in your code and PR description.

### Q: How do I know if my code meets the standards?

**A**: Run `make check` locally. It runs all linters, type checkers, security scanners, and tests.

### Q: What about performance vs. correctness trade-offs?

**A**: Correctness and security always come first. Optimize only after proving correctness and establishing benchmarks.

---

## Next Steps

### For New Contributors

1. Read `.github/copilot-instructions.md` in full
2. Set up your dev environment: `make dev-setup`
3. Review existing code to understand patterns
4. Start with small contributions to learn the system

### For Maintainers

1. Keep instructions aligned with evolving best practices
2. Update repository-specific context as architecture changes
3. Review PRs for compliance with Copilot instructions
4. Provide feedback on Copilot suggestion quality

### For Advanced Configuration

- Split instructions into role-specific files (Agent, Reviewer, Refactorer)
- Add language-specific overlays for TypeScript, Rust, Go, C++
- Create enforcement addendum for restricted operations
- Integrate with enterprise Copilot policies

---

## Version History

- **v1.0** (2026-01-16): Initial production-grade instructions
  - Senior Principal Engineer + Systems Architect + AI Reliability Officer role
  - 12 core instruction sections
  - Repository-specific context for FRED Supreme Litigation OS
  - Fail-closed engineering and security-by-default principles

---

## Resources

- **GitHub Copilot Docs**: https://docs.github.com/en/copilot
- **Repository Guidelines**: See `CONTRIBUTING.md`
- **CI/CD Guide**: See `CI_CD_GUIDE.md`
- **Development Protocols**: See `docs/development_protocols.md`

---

**This is a living document. Keep it updated as the repository and team evolve.**
