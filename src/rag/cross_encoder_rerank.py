"""Cross-encoder reranking: the quality knob for production RAG.

Interview goal: a retriever is a bi-encoder -- query and doc are
embedded INDEPENDENTLY, then scored by dot product. Fast, but leaves
quality on the table because the two encodings never see each other.

A reranker is a CROSS-encoder -- query and doc are concatenated and
scored jointly by a transformer. Much slower, but far more accurate.
The standard pipeline is a 2-stage cascade:

    retriever (top 100)  ->  cross-encoder reranker (top 5)  ->  LLM

Real systems use MiniLM cross-encoders or bge-reranker. Here we mock
the cross-encoder with a keyword overlap scorer so the demo is offline.
The pipeline structure is what matters.
"""

from __future__ import annotations

import re
from typing import Callable, List, Tuple


def tokens(s: str) -> set:
    return set(re.findall(r"[a-z]+", s.lower()))


# ---------- stage 1: fast retriever (bi-encoder stub) ----------

def retrieve_fast(query: str, docs: List[str], top_k: int) -> List[int]:
    """Cheap keyword-overlap retriever standing in for a dense bi-encoder."""
    q = tokens(query)
    scored = [(len(q & tokens(d)), i) for i, d in enumerate(docs)]
    scored.sort(reverse=True)
    return [i for _, i in scored[:top_k]]


# ---------- stage 2: cross-encoder reranker ----------

def cross_encoder_score(query: str, doc: str) -> float:
    """Stub for a real cross-encoder. Rewards full bigram matches over
    unigram overlap, so it ranks doc with the exact phrase higher."""
    q_tokens = re.findall(r"[a-z]+", query.lower())
    d_text = " " + doc.lower() + " "
    uni = sum(1 for t in q_tokens if f" {t} " in d_text)
    bi = sum(
        1 for i in range(len(q_tokens) - 1)
        if f" {q_tokens[i]} {q_tokens[i+1]} " in d_text
    )
    # Reward exact-phrase hits strongly.
    return uni + 3.0 * bi


def rerank(query: str, candidates: List[int], docs: List[str], top_k: int) -> List[int]:
    scored = [(cross_encoder_score(query, docs[i]), i) for i in candidates]
    scored.sort(reverse=True)
    return [i for _, i in scored[:top_k]]


# ---------- evaluation ----------

def hit_at_k(ranked: List[int], gold: int, k: int) -> float:
    return float(gold in ranked[:k])


def main():
    docs = [
        "Transformers attend over all token positions via self-attention.",
        "BM25 ranks documents by term frequency and inverse document frequency.",
        "RAG grounds an LLM in retrieved documents at query time.",
        "A bi-encoder computes query and document embeddings independently.",
        "A cross-encoder scores query and document jointly in one forward pass.",
        "Cross-encoders are slower but produce higher quality rankings than bi-encoders.",
        "Dense retrieval uses vector similarity search.",
        "FAISS is a library for efficient similarity search on dense vectors.",
        "Tokenizers split text into subword units.",
        "KV caching speeds up autoregressive decoding.",
    ]
    q = "What is a cross-encoder reranker and why is it slower than a bi-encoder?"
    gold = 5  # the doc that actually answers the question

    top_n = 5
    stage1 = retrieve_fast(q, docs, top_k=top_n)
    stage2 = rerank(q, stage1, docs, top_k=3)

    print(f"Query: {q}\n")
    print(f"Stage 1 retrieval (top {top_n}) -> {stage1}")
    for i in stage1:
        print(f"  {i}: {docs[i]}")
    print(f"\nStage 2 reranked (top 3)  -> {stage2}")
    for i in stage2:
        print(f"  {i}: {docs[i]}")

    print()
    print(f"Hit@1 retriever : {hit_at_k(stage1, gold, 1):.0f}")
    print(f"Hit@1 reranker  : {hit_at_k(stage2, gold, 1):.0f}")
    print(f"Hit@3 retriever : {hit_at_k(stage1, gold, 3):.0f}")
    print(f"Hit@3 reranker  : {hit_at_k(stage2, gold, 3):.0f}")

    print("\nCost model reminder:")
    print("  Retrieving 100 candidates with a bi-encoder is ~1 inference call.")
    print("  Reranking them with a cross-encoder is 100 inference calls.")
    print("  This is why rerank is used on 50-200 candidates, not millions.")


if __name__ == "__main__":
    main()
