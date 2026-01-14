#!/usr/bin/env python3
"""System health check and validation script.

Validates all components of the FRED Supreme Litigation OS.
"""

import sys
import importlib
from pathlib import Path
from typing import List, Tuple, Dict

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


class HealthChecker:
    """Comprehensive system health checker."""

    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        self.warnings: List[str] = []

    def check_imports(self) -> bool:
        """Check that all core modules can be imported."""
        print(f"\n{BLUE}=== Checking Module Imports ==={RESET}")

        modules = [
            "core.exceptions",
            "core.constants",
            "ai.nlp_document_processor",
            "ai.evidence_llm_analyzer",
        ]

        all_ok = True
        for module in modules:
            try:
                importlib.import_module(module)
                self._log_success(f"Import {module}")
            except Exception as e:
                self._log_failure(f"Import {module}", str(e))
                all_ok = False

        return all_ok

    def check_directories(self) -> bool:
        """Check that required directories exist."""
        print(f"\n{BLUE}=== Checking Directory Structure ==={RESET}")

        required_dirs = [
            "ai",
            "core",
            "config",
            "cli",
            "src",
            "tests",
            "docs",
            "modules",
        ]

        all_ok = True
        for dirname in required_dirs:
            dirpath = Path(dirname)
            if dirpath.exists() and dirpath.is_dir():
                self._log_success(f"Directory {dirname}")
            else:
                self._log_failure(f"Directory {dirname}", "Not found")
                all_ok = False

        return all_ok

    def check_init_files(self) -> bool:
        """Check that __init__.py files exist in packages."""
        print(f"\n{BLUE}=== Checking Package Initialization ==={RESET}")

        packages = [
            "ai",
            "core",
            "config",
            "cli",
            "src",
            "tests",
            "modules",
        ]

        all_ok = True
        for package in packages:
            init_file = Path(package) / "__init__.py"
            if init_file.exists():
                self._log_success(f"Package {package}")
            else:
                self._log_warning(f"Package {package}", "Missing __init__.py")
                all_ok = False

        return all_ok

    def check_python_syntax(self) -> bool:
        """Check Python files for syntax errors."""
        print(f"\n{BLUE}=== Checking Python Syntax ==={RESET}")

        import py_compile
        import glob

        python_files = []
        for pattern in ["ai/*.py", "core/*.py", "src/*.py"]:
            python_files.extend(glob.glob(pattern))

        all_ok = True
        errors = 0
        for filepath in python_files:
            try:
                py_compile.compile(filepath, doraise=True)
            except py_compile.PyCompileError as e:
                self._log_failure(f"Syntax {filepath}", str(e))
                all_ok = False
                errors += 1

        if all_ok:
            self._log_success(f"All {len(python_files)} Python files valid")
        else:
            print(f"{RED}✗ {errors} files with syntax errors{RESET}")

        return all_ok

    def check_dependencies(self) -> bool:
        """Check that key dependencies are available."""
        print(f"\n{BLUE}=== Checking Dependencies ==={RESET}")

        required = [
            "dataclasses",
            "typing",
            "json",
            "pathlib",
            "logging",
        ]

        optional = [
            "transformers",
            "torch",
            "numpy",
            "pandas",
        ]

        all_ok = True
        for module in required:
            try:
                importlib.import_module(module)
                self._log_success(f"Required: {module}")
            except ImportError:
                self._log_failure(f"Required: {module}", "Not installed")
                all_ok = False

        for module in optional:
            try:
                importlib.import_module(module)
                self._log_success(f"Optional: {module}")
            except ImportError:
                self._log_warning(f"Optional: {module}", "Not installed")

        return all_ok

    def _log_success(self, item: str):
        """Log a successful check."""
        print(f"{GREEN}✓{RESET} {item}")
        self.results.append((item, True, ""))

    def _log_failure(self, item: str, reason: str):
        """Log a failed check."""
        print(f"{RED}✗{RESET} {item}: {reason}")
        self.results.append((item, False, reason))

    def _log_warning(self, item: str, reason: str):
        """Log a warning."""
        print(f"{YELLOW}⚠{RESET} {item}: {reason}")
        self.warnings.append(f"{item}: {reason}")

    def print_summary(self):
        """Print summary of all checks."""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}=== Health Check Summary ==={RESET}")
        print(f"{BLUE}{'='*60}{RESET}")

        passed = sum(1 for _, ok, _ in self.results if ok)
        failed = len(self.results) - passed

        print(f"\nTotal Checks: {len(self.results)}")
        print(f"{GREEN}Passed: {passed}{RESET}")
        print(f"{RED}Failed: {failed}{RESET}")
        print(f"{YELLOW}Warnings: {len(self.warnings)}{RESET}")

        if failed == 0:
            print(f"\n{GREEN}{'='*60}")
            print(f"✓✓✓ SYSTEM HEALTHY - ALL CHECKS PASSED ✓✓✓")
            print(f"{'='*60}{RESET}\n")
            return 0
        else:
            print(f"\n{RED}{'='*60}")
            print(f"✗✗✗ SYSTEM ISSUES DETECTED ✗✗✗")
            print(f"{'='*60}{RESET}\n")
            return 1

    def run_all_checks(self) -> int:
        """Run all health checks."""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}FRED Supreme Litigation OS - Health Check{RESET}")
        print(f"{BLUE}{'='*60}{RESET}")

        checks = [
            self.check_directories,
            self.check_init_files,
            self.check_imports,
            self.check_python_syntax,
            self.check_dependencies,
        ]

        for check in checks:
            try:
                check()
            except Exception as e:
                print(f"{RED}✗ Check failed with exception: {e}{RESET}")

        return self.print_summary()


def main():
    """Main entry point."""
    checker = HealthChecker()
    exit_code = checker.run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
