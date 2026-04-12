from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalExample:
    query: str
    gold_doc_id: str
    ranked_doc_ids: list[str]


def precision_at_k(example: RetrievalExample, k: int) -> float:
    k = max(1, k)
    topk = example.ranked_doc_ids[:k]
    return 1.0 / k if example.gold_doc_id in topk else 0.0


def recall_at_k(example: RetrievalExample, k: int) -> float:
    k = max(1, k)
    topk = example.ranked_doc_ids[:k]
    return 1.0 if example.gold_doc_id in topk else 0.0


def mrr(example: RetrievalExample) -> float:
    for i, doc_id in enumerate(example.ranked_doc_ids, start=1):
        if doc_id == example.gold_doc_id:
            return 1.0 / i
    return 0.0


def dcg(rels: list[int]) -> float:
    rels = list(rels)
    if not rels:
        return 0.0
    rels_np = np.asarray(rels, dtype=float)
    discounts = np.log2(np.arange(2, len(rels_np) + 2))
    gains = (2 ** rels_np - 1) / discounts
    return float(np.sum(gains))


def ndcg_for_single_relevant(example: RetrievalExample, k: int) -> float:
    k = max(1, k)
    topk = example.ranked_doc_ids[:k]
    rels = [1 if doc_id == example.gold_doc_id else 0 for doc_id in topk]
    ideal = [1] + [0] * (len(rels) - 1)
    denom = dcg(ideal)
    return dcg(rels) / denom if denom else 0.0


def aggregate_metrics(examples: list[RetrievalExample], k: int = 5):
    if not examples:
        return {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "mrr": 0.0,
            "ndcg@k": 0.0,
        }

    p = float(np.mean([precision_at_k(ex, k) for ex in examples]))
    r = float(np.mean([recall_at_k(ex, k) for ex in examples]))
    m = float(np.mean([mrr(ex) for ex in examples]))
    n = float(np.mean([ndcg_for_single_relevant(ex, k) for ex in examples]))

    return {
        "precision@k": p,
        "recall@k": r,
        "mrr": m,
        "ndcg@k": n,
    }
