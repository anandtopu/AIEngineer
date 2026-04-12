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
        Path("src/rag/embeddings_knn_search.py"),
        Path("src/rag/chunking_strategies.py"),
        Path("src/advanced/bpe_tokenizer.py"),
        Path("src/advanced/llm_sampling.py"),
        Path("src/advanced/quantization_int8.py"),
        Path("src/advanced/class_imbalance.py"),
        Path("src/advanced/ab_test_significance.py"),
        Path("src/deep_learning/attention_from_scratch.py"),
        Path("src/deep_learning/transformer_encoder_torch.py"),
        Path("src/deep_learning/lora_from_scratch.py"),
        Path("src/deep_learning/autograd_micro.py"),
        Path("src/deep_learning/rope_positional.py"),
        Path("src/deep_learning/kv_cache.py"),
        Path("src/deep_learning/speculative_decoding.py"),
        Path("src/deep_learning/dpo_from_scratch.py"),
        Path("src/deep_learning/knowledge_distillation.py"),
        Path("src/deep_learning/clip_dual_encoder.py"),
        Path("src/rag/hybrid_search_bm25.py"),
        Path("src/rag/cross_encoder_rerank.py"),
        Path("src/rag/semantic_cache.py"),
        Path("src/rag/ragas_style_eval.py"),
        Path("src/llm/agent_react_loop.py"),
        Path("src/llm/prompt_patterns.py"),
        Path("src/llm/structured_output.py"),
        Path("src/llm/streaming_generator.py"),
        Path("src/llm/guardrails.py"),
        Path("src/llm/token_economics.py"),
        Path("src/llm/observability_tracing.py"),
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
        Path("projects/06_mini_llm_eval/run.py"),
        Path("projects/07_agent_tool_use/run.py"),
        Path("projects/08_hybrid_rag/run.py"),
        Path("projects/09_llm_gateway/run.py"),
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
