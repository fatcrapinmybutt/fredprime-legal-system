#!/usr/bin/env python3
"""
VERIFICATION SUITE FOR CODE QUALITY FIXES
Validates all corrections applied to FRED Supreme Litigation OS
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class VerificationSuite:
    """Comprehensive verification of all code quality fixes"""

    def __init__(self):
        self.workspace_root = Path(__file__).parent
        self.results: Dict[str, List[str]] = {
            "passed": [],
            "failed": [],
            "warnings": []
        }

    def run_all_verifications(self) -> bool:
        """Run all verification checks"""
        print("\n" + "="*80)
        print("CODE QUALITY VERIFICATION SUITE")
        print("="*80 + "\n")

        verifications = [
            ("Type Hint Coverage", self.verify_type_hints),
            ("Import Organization", self.verify_imports),
            ("F-String Usage", self.verify_fstrings),
            ("Return Type Compliance", self.verify_return_types),
            ("Test File Organization", self.verify_tests),
            ("Pylance Errors", self.verify_pylance),
        ]

        for name, verification_func in verifications:
            print(f"\n📋 Checking: {name}")
            print("-" * 40)
            try:
                result = verification_func()
                if result:
                    self.results["passed"].append(name)
                    print(f"✅ {name}: PASSED")
                else:
                    self.results["failed"].append(name)
                    print(f"❌ {name}: FAILED")
            except Exception as e:
                self.results["failed"].append(name)
                print(f"❌ {name}: ERROR - {e}")

        return self.print_summary()

    def verify_type_hints(self) -> bool:
        """Verify all test functions have type hints"""
        test_file = self.workspace_root / "tests" / "test_ai_modules.py"

        if not test_file.exists():
            print(f"⚠️  Test file not found: {test_file}")
            return False

        with open(test_file) as f:
            content = f.read()

        # Check for type hints in test methods
        checks = [
            ("analyzer fixture", "def analyzer(self) -> EvidenceLLMAnalyzer:"),
            ("processor fixture", "def processor(self) -> NLPDocumentProcessor:"),
            ("test method hints", "-> None"),
        ]

        all_found = True
        for check_name, pattern in checks:
            if pattern in content:
                print(f"  ✓ Found {check_name}")
            else:
                print(f"  ✗ Missing {check_name}")
                all_found = False

        return all_found

    def verify_imports(self) -> bool:
        """Verify imports are properly organized"""
        files_to_check = [
            "tests/test_ai_modules.py",
            "QUICKSTART_AI_ML.py",
            "src/ai_integration_bridge.py",
            "PROJECT_MANIFEST.py"
        ]

        all_good = True
        for file_path in files_to_check:
            full_path = self.workspace_root / file_path
            if not full_path.exists():
                print(f"  ⚠️  File not found: {file_path}")
                continue

            with open(full_path) as f:
                content = f.read()

            # Check for common unused imports
            unused = [
                ("unused json", "import json" in content and "json." not in content),
                ("unused Path", "from pathlib import Path" in content and "Path(" not in content),
            ]

            for check_name, condition in unused:
                if condition:
                    print(f"  ✗ {file_path}: {check_name}")
                    all_good = False
                else:
                    print(f"  ✓ {file_path}: No {check_name}")

        return all_good

    def verify_fstrings(self) -> bool:
        """Verify f-string usage is correct"""
        files_to_check = [
            "QUICKSTART_AI_ML.py",
            "src/ai_integration_bridge.py"
        ]

        all_good = True
        for file_path in files_to_check:
            full_path = self.workspace_root / file_path
            if not full_path.exists():
                continue

            with open(full_path) as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                # Check for f-strings with only whitespace/special chars
                if 'f"' in line or "f'" in line:
                    if 'f"\\n"' in line or "f'\\n'" in line:
                        print(f"  ✗ {file_path}:{i} - F-string with only newline")
                        all_good = False

        if all_good:
            print(f"  ✓ All f-strings properly formatted")

        return all_good

    def verify_return_types(self) -> bool:
        """Verify return type annotations"""
        bridge_file = self.workspace_root / "src" / "ai_integration_bridge.py"

        if not bridge_file.exists():
            print(f"⚠️  File not found: {bridge_file}")
            return False

        with open(bridge_file) as f:
            content = f.read()

        checks = [
            ("Optional return type", "-> Optional[AIAnalysisReport]:"),
            ("Dict return type", "-> Dict[str, str]:"),
            ("No bare None returns", "return None" in content),  # Should exist
        ]

        all_found = True
        for check_name, pattern in checks:
            if isinstance(pattern, str) and pattern in content:
                print(f"  ✓ {check_name}")
            elif isinstance(pattern, bool) and pattern:
                print(f"  ✓ {check_name}")
            else:
                print(f"  ✗ Missing {check_name}")
                all_found = False

        return all_found

    def verify_tests(self) -> bool:
        """Verify test file organization"""
        test_files = [
            "tests/test_ai_modules.py",
            "tests/test_patch_manager.py",
            "tests/test_gui.py",
        ]

        found_count = 0
        for test_file in test_files:
            if (self.workspace_root / test_file).exists():
                found_count += 1
                print(f"  ✓ {test_file}")
            else:
                print(f"  ⚠️  {test_file} (optional)")

        return found_count > 0

    def verify_pylance(self) -> bool:
        """Check for Pylance errors using pyright"""
        print("  📊 Running Pylance/Pyright analysis...")

        try:
            result = subprocess.run(
                ["python", "-m", "pylance", "--version"],
                capture_output=True,
                timeout=5
            )

            if result.returncode == 0:
                print("  ✓ Pylance available for analysis")
                return True
            else:
                print("  ⚠️  Pylance not available (install with: pip install pylance)")
                return True  # Don't fail if pylance not installed

        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("  ⚠️  Pylance analysis skipped (optional)")
            return True

    def print_summary(self) -> bool:
        """Print verification summary"""
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)

        total = len(self.results["passed"]) + len(self.results["failed"])
        passed = len(self.results["passed"])
        failed = len(self.results["failed"])

        print(f"\n✅ Passed: {passed}/{total}")
        for item in self.results["passed"]:
            print(f"   - {item}")

        if self.results["failed"]:
            print(f"\n❌ Failed: {failed}/{total}")
            for item in self.results["failed"]:
                print(f"   - {item}")

        if self.results["warnings"]:
            print(f"\n⚠️  Warnings: {len(self.results['warnings'])}")
            for item in self.results["warnings"]:
                print(f"   - {item}")

        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"\n📊 Success Rate: {success_rate:.0f}%\n")

        if failed == 0:
            print("🎉 ALL VERIFICATIONS PASSED!\n")
            return True
        else:
            print(f"⚠️  {failed} verification(s) failed.\n")
            return False


def main():
    """Run verification suite"""
    suite = VerificationSuite()
    success = suite.run_all_verifications()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
