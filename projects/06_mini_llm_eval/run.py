"""Mini LLM evaluation harness — exact match, token F1, and a judge.

Interview goal: walk through the building blocks of an LLM eval system
without depending on a real API. We define:
  - a deterministic mock LLM that picks one of several canned responses
  - reference-based metrics (exact match, token F1, ROUGE-L)
  - a "model-as-judge" stub showing how to wire LLM-graded eval
  - a small task suite with stratified slices and per-slice scores

Real harnesses (lm-eval-harness, OpenAI evals, Anthropic evals) follow
exactly this shape: dataset -> generator -> grader -> aggregator.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Callable, List


# ----------------------- metrics -----------------------

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks or not g_toks:
        return 0.0
    common = {}
    for t in p_toks:
        common[t] = min(p_toks.count(t), g_toks.count(t))
    n_common = sum(common.values()) // 1  # already counted carefully
    # Recompute with set logic for clarity:
    overlap = sum(min(p_toks.count(t), g_toks.count(t)) for t in set(p_toks) & set(g_toks))
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    """Longest common subsequence based F1 — the heart of ROUGE-L."""
    a = normalize(pred).split()
    b = normalize(gold).split()
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = (
                dp[i][j] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
            )
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r)


def llm_judge(pred: str, gold: str) -> float:
    """Stub LLM-as-judge: gives 1.0 if normalized gold is a substring of pred."""
    return float(normalize(gold) in normalize(pred))


# ----------------------- task & mock LLM -----------------------

@dataclass
class Example:
    qid: str
    question: str
    answer: str
    slice: str  # e.g. "factoid" / "math" / "reasoning"


@dataclass
class EvalResult:
    metric: str
    overall: float
    by_slice: dict = field(default_factory=dict)


def mock_llm(question: str) -> str:
    """A deterministic 'model' — good enough to exercise the harness."""
    table = {
        "Capital of France?": "The capital of France is Paris.",
        "2 + 2 = ?": "4",
        "Who wrote Hamlet?": "William Shakespeare",
        "sqrt(81)?": "Nine",  # intentionally written out, not '9'
        "Largest planet?": "Jupiter is the largest planet in our solar system.",
        "Speed of light (m/s)?": "approximately 3e8 m/s",
    }
    return table.get(question, "I don't know.")


def run_metric(metric: Callable[[str, str], float], examples: List[Example],
               predictions: List[str], name: str) -> EvalResult:
    scores = [metric(p, e.answer) for p, e in zip(predictions, examples)]
    by_slice = {}
    for e, s in zip(examples, scores):
        by_slice.setdefault(e.slice, []).append(s)
    return EvalResult(
        metric=name,
        overall=statistics.mean(scores),
        by_slice={k: statistics.mean(v) for k, v in by_slice.items()},
    )


def main():
    suite = [
        Example("q1", "Capital of France?",     "Paris",            "factoid"),
        Example("q2", "2 + 2 = ?",              "4",                "math"),
        Example("q3", "Who wrote Hamlet?",      "Shakespeare",      "factoid"),
        Example("q4", "sqrt(81)?",              "9",                "math"),
        Example("q5", "Largest planet?",        "Jupiter",          "factoid"),
        Example("q6", "Speed of light (m/s)?",  "299792458",        "math"),
    ]
    preds = [mock_llm(e.question) for e in suite]

    print("=== Predictions ===")
    for e, p in zip(suite, preds):
        print(f"  [{e.slice:7s}] {e.question:25s} -> {p!r}  (gold={e.answer!r})")

    metrics = {
        "exact_match": exact_match,
        "token_f1":    token_f1,
        "rouge_l":     rouge_l,
        "llm_judge":   llm_judge,
    }

    print("\n=== Aggregate scores ===")
    print(f"  {'metric':<12s} {'overall':>8s}   per-slice")
    for name, fn in metrics.items():
        r = run_metric(fn, suite, preds, name)
        sl = "  ".join(f"{k}={v:.2f}" for k, v in sorted(r.by_slice.items()))
        print(f"  {name:<12s} {r.overall:>8.2f}   {sl}")

    print("\nNotes:")
    print("  - exact_match is harsh: 'Nine' != '9' even though both are right")
    print("  - token_f1 / rouge_l reward partial overlap")
    print("  - llm_judge handles paraphrase ('Paris' inside a longer answer)")
    print("  - real harnesses run thousands of examples and report CIs")


if __name__ == "__main__":
    main()
