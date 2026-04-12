"""Validation script to check repo setup and run key examples."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


def check_imports() -> list[str]:
    """Check that key dependencies are importable."""
    required = ["numpy", "pandas", "sklearn", "torch", "torchvision"]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
            print(f"  {pkg}: OK")
        except ImportError:
            missing.append(pkg)
            print(f"  {pkg}: MISSING")
    return missing


def run_script(path: Path) -> bool:
    """Run a Python script and return success status."""
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            print(f"  {path}: OK")
            return True
        else:
            print(f"  {path}: FAIL (exit code {result.returncode})")
            print(result.stderr[:200])
            return False
    except subprocess.TimeoutExpired:
        print(f"  {path}: TIMEOUT")
        return False
    except Exception as e:
        print(f"  {path}: ERROR ({e})")
        return False


def main():
    print("=== AI Engineer Interview Prep - Validation ===\n")

    # Check imports
    print("Checking dependencies...")
    missing = check_imports()
    if missing:
        print(f"\nMissing packages: {missing}")
        print("Run: pip install -r requirements.txt")
        return 1

    # Run quick tests
    print("\nRunning quick validation tests...")
    tests = [
        Path("src/basics/linear_regression_numpy.py"),
        Path("src/basics/metrics_from_scratch.py"),
        Path("src/sql/sql_practice_sqlite.py"),
        Path("src/sql/window_functions_practice_sqlite.py"),
        Path("src/rag/tfidf_rag_demo.py"),
    ]

    failed = 0
    for test in tests:
        if not run_script(test):
            failed += 1

    print("\nRunning project examples (may take longer)...")
    projects = [
        Path("projects/01_churn_prediction/train.py"),
        Path("projects/02_rag_baseline/run.py"),
        Path("projects/03_monitoring_drift/drift_check.py"),
        Path("projects/04_ranking_baseline/train.py"),
        Path("projects/05_time_series/baseline.py"),
    ]

    for proj in projects:
        if not run_script(proj):
            failed += 1

    print("\n=== Summary ===")
    if failed == 0:
        print("All checks passed! Repository is ready.")
        return 0
    else:
        print(f"{failed} test(s) failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
