"""Hybrid search: BM25 + dense + Reciprocal Rank Fusion.

Interview goal: production RAG systems almost never use dense retrieval
alone. BM25 is great at exact matches (names, numbers, rare words);
dense embeddings are great at semantic similarity. Fusing them beats
either on its own. The standard fusion is Reciprocal Rank Fusion (RRF):

    RRF_score(d) = sum over rankers r of  1 / (k + rank_r(d))

with k typically 60. It needs no score calibration -- only the ranks --
which makes it the de-facto industry baseline.

This file implements BM25 from scratch, a tiny dense retriever, and
combines them with RRF. We then measure MRR of each ranker to show
that the fusion wins.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, List


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------- BM25 ----------

class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        self.df: Dict[str, int] = Counter()
        self.tf: List[Counter] = []
        for d in self.docs:
            self.tf.append(Counter(d))
            for t in set(d):
                self.df[t] += 1
        self.k1 = k1
        self.b = b

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query: str) -> List[float]:
        q = tokenize(query)
        scores = [0.0] * self.N
        for i, doc in enumerate(self.docs):
            dl = len(doc)
            for term in q:
                if term not in self.tf[i]:
                    continue
                f = self.tf[i][term]
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += self.idf(term) * num / den
        return scores

    def rank(self, query: str) -> List[int]:
        s = self.score(query)
        return sorted(range(self.N), key=lambda i: -s[i])


# ---------- tiny dense retriever (hashing trick embedding) ----------

DIM = 64


def embed(text: str) -> List[float]:
    vec = [0.0] * DIM
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0 if (h >> 8) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n > 0 else vec


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


class DenseRetriever:
    def __init__(self, docs: List[str]):
        self.index = [embed(d) for d in docs]

    def rank(self, query: str) -> List[int]:
        q = embed(query)
        scored = sorted(range(len(self.index)), key=lambda i: -cosine(self.index[i], q))
        return scored


# ---------- reciprocal rank fusion ----------

def rrf(rankings: List[List[int]], k: int = 60, top_k: int = 10) -> List[int]:
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])[:top_k]


def mrr(rankings: List[List[int]], gold: List[int]) -> float:
    total = 0.0
    for r, g in zip(rankings, gold):
        if g in r:
            total += 1.0 / (r.index(g) + 1)
    return total / len(gold)


def main():
    docs = [
        "FAISS is a library for fast approximate nearest neighbor search on dense vectors.",
        "BM25 is a bag-of-words ranking function used by classical search engines.",
        "A hybrid retriever combines lexical and semantic signals.",
        "RRF merges multiple rankings using only the document ranks.",
        "Dense retrievers encode text into fixed-size vectors with a neural model.",
        "The BERT model is a bidirectional transformer encoder trained on masked LM.",
        "Tokenization with BPE splits words into subword units.",
        "Vector databases index billions of embeddings for similarity search.",
        "Learning to rank trains a model to order search results by relevance.",
        "Retrieval augmented generation grounds LLM answers in retrieved documents.",
    ]
    queries = [
        "What is BM25?",
        "How does FAISS help retrieval?",
        "Explain RRF in simple terms.",
        "How does RAG avoid hallucinations?",
    ]
    gold = [1, 0, 3, 9]  # index of best doc per query

    bm25 = BM25(docs)
    dense = DenseRetriever(docs)

    bm25_rankings = [bm25.rank(q) for q in queries]
    dense_rankings = [dense.rank(q) for q in queries]
    fused = [rrf([b, d]) for b, d in zip(bm25_rankings, dense_rankings)]

    print("MRR comparison:")
    print(f"  BM25 only : {mrr(bm25_rankings, gold):.3f}")
    print(f"  Dense only: {mrr(dense_rankings, gold):.3f}")
    print(f"  RRF fused : {mrr(fused, gold):.3f}")

    print("\nExample -- query 'What is BM25?' top-3:")
    for name, r in [("BM25", bm25_rankings[0]), ("Dense", dense_rankings[0]), ("Fused", fused[0])]:
        top3 = r[:3]
        print(f"  {name:>5s}: {top3}  -> {docs[top3[0]][:60]}")

    print("\nWhy this matters:")
    print("  - BM25 nails rare/exact-word queries ('BM25', 'FAISS').")
    print("  - Dense wins on paraphrase ('avoid hallucinations').")
    print("  - RRF captures both without needing score calibration.")


if __name__ == "__main__":
    main()
