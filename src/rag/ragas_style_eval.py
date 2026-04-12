"""RAG evaluation metrics (RAGAS-style) without external dependencies.

Interview goal: know the four canonical RAG metrics and what each of
them ACTUALLY measures. RAGAS popularized this taxonomy:

  1. CONTEXT PRECISION   -- Of the retrieved chunks, how many are relevant?
                            Isolates the retriever.
  2. CONTEXT RECALL      -- Of the facts needed to answer, how many are in
                            the retrieved context?
  3. FAITHFULNESS        -- Of the claims in the answer, how many are
                            supported by the retrieved context?
                            Detects hallucination.
  4. ANSWER RELEVANCY    -- Does the answer actually address the question?

Real RAGAS uses an LLM judge. Here we build deterministic proxies
(sentence-level word overlap) so the script runs offline and the math
is obvious. The interpretation is identical.

Diagnostic rule of thumb:
  faithfulness high + relevancy LOW  -> retrieval is bringing junk
  faithfulness LOW + relevancy high  -> LLM is hallucinating
  precision high + recall LOW        -> you need more / bigger chunks
  precision LOW + recall high        -> chunks are too big / noisy
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


def sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def words(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


def overlap(a: str, b: str, tau: float = 0.3) -> bool:
    """Does sentence `a` have enough content overlap with sentence `b`?"""
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa) >= tau


@dataclass
class RAGSample:
    question: str
    contexts: List[str]       # retrieved chunks
    answer: str               # what the LLM produced
    ground_truth: str         # the human reference answer


def context_precision(sample: RAGSample) -> float:
    """Fraction of retrieved chunks that share content with the ground truth."""
    relevant = [1 if overlap(c, sample.ground_truth) else 0 for c in sample.contexts]
    if not relevant:
        return 0.0
    return sum(relevant) / len(relevant)


def context_recall(sample: RAGSample) -> float:
    """Fraction of ground-truth sentences covered by at least one retrieved chunk."""
    gt_sents = sentences(sample.ground_truth)
    if not gt_sents:
        return 0.0
    covered = 0
    joined = " ".join(sample.contexts)
    for gs in gt_sents:
        if overlap(gs, joined):
            covered += 1
    return covered / len(gt_sents)


def faithfulness(sample: RAGSample) -> float:
    """Fraction of answer sentences that are supported by the retrieved context."""
    ans_sents = sentences(sample.answer)
    if not ans_sents:
        return 0.0
    joined = " ".join(sample.contexts)
    supported = sum(1 for s in ans_sents if overlap(s, joined))
    return supported / len(ans_sents)


def answer_relevancy(sample: RAGSample) -> float:
    """How much of the question's content is addressed by the answer?"""
    qw = words(sample.question)
    aw = words(sample.answer)
    if not qw:
        return 0.0
    # Remove stop-ish words: 1-2 chars
    qw = {w for w in qw if len(w) > 2}
    if not qw:
        return 0.0
    return len(qw & aw) / len(qw)


def evaluate(samples: List[RAGSample]) -> dict:
    metrics = {"context_precision": [], "context_recall": [],
               "faithfulness": [], "answer_relevancy": []}
    for s in samples:
        metrics["context_precision"].append(context_precision(s))
        metrics["context_recall"].append(context_recall(s))
        metrics["faithfulness"].append(faithfulness(s))
        metrics["answer_relevancy"].append(answer_relevancy(s))
    return {k: sum(v) / len(v) for k, v in metrics.items()}


def main():
    samples = [
        # 1. Good retrieval, good answer.
        RAGSample(
            question="What tokenizer does GPT use?",
            contexts=[
                "GPT models use byte pair encoding (BPE) tokenization.",
                "BERT uses a different WordPiece tokenizer.",
            ],
            answer="GPT models use BPE tokenization.",
            ground_truth="GPT uses byte pair encoding (BPE).",
        ),
        # 2. Good retrieval, but the answer hallucinates an extra fact.
        RAGSample(
            question="Who introduced the Transformer?",
            contexts=[
                "The Transformer was introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al.",
            ],
            answer="The Transformer was introduced by Vaswani et al in 2017, "
                   "and it was invented at Stanford University.",
            ground_truth="The Transformer was introduced by Vaswani et al in 2017.",
        ),
        # 3. Retrieval misses: answer is right but not grounded.
        RAGSample(
            question="What is LoRA?",
            contexts=[
                "BM25 is a classical ranking function.",
                "FAISS is a library for similarity search.",
            ],
            answer="LoRA adds trainable low-rank matrices to a frozen base model.",
            ground_truth="LoRA injects low-rank adapter matrices into a frozen model.",
        ),
    ]
    for i, s in enumerate(samples, 1):
        print(f"\n--- sample {i}: {s.question} ---")
        print(f"  context_precision: {context_precision(s):.2f}")
        print(f"  context_recall   : {context_recall(s):.2f}")
        print(f"  faithfulness     : {faithfulness(s):.2f}")
        print(f"  answer_relevancy : {answer_relevancy(s):.2f}")

    agg = evaluate(samples)
    print(f"\n=== Aggregate over {len(samples)} samples ===")
    for k, v in agg.items():
        print(f"  {k:<18s}: {v:.2f}")

    print("\nInterpretation:")
    print("  - Sample 2: high precision but faithfulness < 1 -> hallucinated 'Stanford'.")
    print("  - Sample 3: faithfulness and precision tank -> retriever failed.")
    print("  - In production, score each metric per slice and alert on regressions.")


if __name__ == "__main__":
    main()
