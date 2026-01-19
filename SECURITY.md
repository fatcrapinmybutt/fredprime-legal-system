# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in DriveOrganizerPro, please report it responsibly.

### Where to Report

- **Email**: security@mbpllc.example (or use GitHub Security Advisories)
- **GitHub**: [Security Advisories](https://github.com/fatcrapinmybutt/fredprime-legal-system/security/advisories/new)

### What to Include

When reporting a vulnerability, please include:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** assessment
4. **Suggested fix** (if you have one)
5. **Your contact information** for follow-up

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 5 business days
- **Status Update**: Every 7 days until resolved
- **Fix Release**: Depends on severity (critical issues prioritized)

### Security Best Practices

When using DriveOrganizerPro:

1. **Backup First**: Always backup important data before organizing
2. **Test in Dry Run**: Use dry run mode to preview changes
3. **Verify Paths**: Double-check source and destination paths
4. **Keep Updated**: Use the latest version for security fixes
5. **Review Logs**: Check logs for unexpected behavior

### Known Security Considerations

#### File System Access

DriveOrganizerPro requires file system access to:
- Read file metadata
- Move files between directories
- Create backup logs

**Mitigation**: 
- Only run on drives you own/control
- Review dry run output before live operations
- Use revert functionality if needed

#### Hash Calculations

MD5/SHA256 hashing is used for duplicate detection:
- File contents are read in chunks
- Hashes are computed locally
- No data is transmitted externally

**Mitigation**:
- Hashing is optional (can be disabled)
- All processing is local

### Out of Scope

The following are **not** considered security vulnerabilities:

- File organization results you don't like (use revert)
- Performance issues
- User error in path selection
- Operating system file permission errors
- Third-party dependency vulnerabilities (report to those projects)

### Responsible Disclosure

We request that security researchers:

1. **Do not** publicly disclose vulnerabilities before a fix is available
2. **Give us** reasonable time to address the issue (typically 90 days)
3. **Work with us** to understand and fix the problem
4. **Act in good faith** - no attacks on production systems

### Recognition

We appreciate security researchers who help us keep DriveOrganizerPro secure. 
With your permission, we'll acknowledge your contribution in our changelog and security advisories.

---

© 2026 MBP LLC. All rights reserved. Powered by Pork™
