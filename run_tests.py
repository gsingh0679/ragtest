#!/usr/bin/env python
"""
Test runner script for RAG Document Chunking project.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py quick        # Run quick tests
    python run_tests.py chunker      # Run chunker tests only
    python run_tests.py loader       # Run loader tests only
    python run_tests.py memory       # Run memory tests only
    python run_tests.py performance  # Run performance tests only
    python run_tests.py setup        # Run setup tests only
"""

import sys
import subprocess
from pathlib import Path


def run_test(test_name):
    """Run a specific test file."""
    test_file = Path("tests") / f"{test_name}.py"
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False

    print(f"\n{'='*80}")
    print(f"Running: {test_name}")
    print(f"{'='*80}\n")

    result = subprocess.run([sys.executable, str(test_file)])
    return result.returncode == 0


def main():
    """Run tests based on command line arguments."""

    if len(sys.argv) < 2 or sys.argv[1] == "all":
        # Run all tests
        print("🧪 Running ALL tests...\n")
        tests = [
            "test_setup",
            "test_document_loader",
            "test_text_chunker",
            "test_memory_usage",
            "test_performance",
        ]
        results = {}
        for test in tests:
            results[test] = run_test(test)

        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}\n")

        passed = sum(1 for v in results.values() if v)
        total = len(results)

        for test, passed_test in results.items():
            status = "✓ PASS" if passed_test else "✗ FAIL"
            print(f"  {status}: {test}")

        print(f"\nTotal: {passed}/{total} passed\n")

        return 0 if passed == total else 1

    elif sys.argv[1] == "quick":
        # Run quick tests (skip performance and memory)
        print("⚡ Running QUICK tests...\n")
        tests = [
            "test_setup",
            "test_document_loader",
            "test_text_chunker",
        ]
        results = {}
        for test in tests:
            results[test] = run_test(test)

        print(f"\n{'='*80}")
        print("QUICK TEST SUMMARY")
        print(f"{'='*80}\n")

        passed = sum(1 for v in results.values() if v)
        total = len(results)

        for test, passed_test in results.items():
            status = "✓ PASS" if passed_test else "✗ FAIL"
            print(f"  {status}: {test}")

        print(f"\nTotal: {passed}/{total} passed\n")

        return 0 if passed == total else 1

    elif sys.argv[1] == "chunker":
        return 0 if run_test("test_text_chunker") else 1

    elif sys.argv[1] == "loader":
        return 0 if run_test("test_document_loader") else 1

    elif sys.argv[1] == "memory":
        return 0 if run_test("test_memory_usage") else 1

    elif sys.argv[1] == "performance":
        return 0 if run_test("test_performance") else 1

    elif sys.argv[1] == "setup":
        return 0 if run_test("test_setup") else 1

    else:
        print(f"Unknown option: {sys.argv[1]}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
