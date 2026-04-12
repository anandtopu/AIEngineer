"""Project 08 -- Production-style hybrid RAG with eval.

End-to-end pipeline:

    query -> BM25 + dense retrieval -> RRF fusion (top 20)
          -> cross-encoder rerank (top 3)
          -> grounded answer generation (extractive)
          -> RAGAS-style faithfulness + relevancy scoring

Every stage is a pure-Python function you can swap for its production
counterpart (FAISS, Cohere rerank, Anthropic API, RAGAS).
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------

CORPUS: List[str] = [
    "LoRA adds trainable low-rank matrices to a frozen base model, training "
    "far fewer parameters than full fine-tuning.",
    "QLoRA combines 4-bit quantization with LoRA adapters so you can fine-tune "
    "large models on a single GPU.",
    "The Transformer architecture was introduced in the 2017 paper "
    "'Attention Is All You Need' by Vaswani et al.",
    "BM25 is a bag-of-words ranking function based on term frequency and "
    "inverse document frequency.",
    "Dense retrievers encode text into fixed-size vectors and use cosine "
    "similarity or inner product for nearest-neighbor search.",
    "Hybrid retrieval fuses lexical and dense rankings, usually with "
    "reciprocal rank fusion (RRF).",
    "Cross-encoder rerankers score a (query, document) pair jointly and are "
    "much more accurate than bi-encoders, at the cost of latency.",
    "RAG grounds an LLM in retrieved chunks to reduce hallucination and keep "
    "answers current.",
    "Faithfulness checks whether every claim in an LLM answer is supported "
    "by the retrieved context.",
    "Semantic caching stores previous answers keyed by embedding similarity "
    "to short-circuit duplicate LLM calls.",
    "KV caching speeds up autoregressive decoding by storing K, V of prior "
    "tokens so each new token only computes one new query.",
    "Speculative decoding accelerates inference losslessly by having a small "
    "draft model propose tokens that a big target model verifies.",
]


# ---------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, docs: List[str], k1=1.5, b=0.75):
        self.docs_tok = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in self.docs_tok) / self.N
        self.df: Dict[str, int] = {}
        self.tf: List[Dict[str, int]] = []
        for d in self.docs_tok:
            tf: Dict[str, int] = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            self.tf.append(tf)
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.k1, self.b = k1, b

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def rank(self, query: str) -> List[int]:
        q = tokenize(query)
        scores = [0.0] * self.N
        for i, doc in enumerate(self.docs_tok):
            dl = len(doc)
            for term in q:
                if term not in self.tf[i]:
                    continue
                f = self.tf[i][term]
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += self.idf(term) * num / den
        return sorted(range(self.N), key=lambda i: -scores[i])


DIM = 64


def embed(text: str) -> List[float]:
    v = [0.0] * DIM
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 8) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def cosine(a, b): return sum(x * y for x, y in zip(a, b))


class DenseIndex:
    def __init__(self, docs: List[str]):
        self.index = [embed(d) for d in docs]

    def rank(self, query: str) -> List[int]:
        q = embed(query)
        return sorted(range(len(self.index)), key=lambda i: -cosine(self.index[i], q))


def rrf(rankings: List[List[int]], k: int = 60) -> List[int]:
    scores: Dict[int, float] = {}
    for r in rankings:
        for rank, doc in enumerate(r):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])


# ---------------------------------------------------------------
# Rerank (cross-encoder stub)
# ---------------------------------------------------------------

def rerank(query: str, candidates: List[int], docs: List[str], top_k: int) -> List[int]:
    q_tokens = tokenize(query)
    q_set = set(q_tokens)

    def score(i: int) -> float:
        d_tokens = tokenize(docs[i])
        d_set = set(d_tokens)
        uni = len(q_set & d_set) / max(1, len(q_set))
        # reward bigram matches
        bi = 0
        d_text = " " + " ".join(d_tokens) + " "
        for a, b in zip(q_tokens, q_tokens[1:]):
            if f" {a} {b} " in d_text:
                bi += 1
        return uni + 0.5 * bi

    return sorted(candidates, key=lambda i: -score(i))[:top_k]


# ---------------------------------------------------------------
# Generator (extractive: build an answer from top chunks)
# ---------------------------------------------------------------

def generate(query: str, top_chunks: List[str]) -> str:
    # Pick the single sentence from the top chunk with the most query-word overlap.
    q = set(tokenize(query))
    best, best_score = "", -1
    for chunk in top_chunks[:2]:
        for s in re.split(r"(?<=[.!?])\s+", chunk):
            score = len(q & set(tokenize(s)))
            if score > best_score:
                best, best_score = s, score
    return best.strip() or top_chunks[0]


# ---------------------------------------------------------------
# Evaluation (RAGAS-style)
# ---------------------------------------------------------------

def words(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


def faithfulness(answer: str, contexts: List[str]) -> float:
    joined = " ".join(contexts)
    ans_sents = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s]
    if not ans_sents:
        return 0.0
    supported = 0
    for s in ans_sents:
        aw = words(s)
        jw = words(joined)
        if aw and len(aw & jw) / len(aw) >= 0.3:
            supported += 1
    return supported / len(ans_sents)


def context_precision(contexts: List[str], gold: str) -> float:
    gw = words(gold)
    if not contexts:
        return 0.0
    return sum(1 for c in contexts if len(words(c) & gw) / max(1, len(words(c))) >= 0.1) / len(contexts)


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------

@dataclass
class Query:
    text: str
    gold_contains: str


def run_pipeline(q: Query, bm25: BM25, dense: DenseIndex, docs: List[str]):
    bm_rank = bm25.rank(q.text)[:20]
    dn_rank = dense.rank(q.text)[:20]
    fused = rrf([bm_rank, dn_rank])[:20]
    reranked = rerank(q.text, fused, docs, top_k=3)
    top_chunks = [docs[i] for i in reranked]
    answer = generate(q.text, top_chunks)
    faith = faithfulness(answer, top_chunks)
    prec = context_precision(top_chunks, q.gold_contains)
    return {
        "answer": answer,
        "top_ids": reranked,
        "faithfulness": faith,
        "context_precision": prec,
    }


def main():
    bm25 = BM25(CORPUS)
    dense = DenseIndex(CORPUS)
    queries = [
        Query("What is LoRA?",
              "LoRA adds trainable low-rank matrices"),
        Query("How does hybrid retrieval combine signals?",
              "reciprocal rank fusion"),
        Query("Why do production systems use a cross-encoder reranker?",
              "score a (query, document) pair jointly"),
        Query("How does speculative decoding speed up inference?",
              "draft model"),
    ]
    total = {"faithfulness": 0.0, "context_precision": 0.0}
    for q in queries:
        r = run_pipeline(q, bm25, dense, CORPUS)
        print(f"\nQ: {q.text}")
        print(f"  top ids: {r['top_ids']}")
        print(f"  answer : {r['answer']}")
        print(f"  faithfulness     : {r['faithfulness']:.2f}")
        print(f"  context_precision: {r['context_precision']:.2f}")
        total["faithfulness"] += r["faithfulness"]
        total["context_precision"] += r["context_precision"]

    print("\n=== Aggregate ===")
    n = len(queries)
    for k, v in total.items():
        print(f"  {k:<20s} {v/n:.2f}")

    print("\nProduction swap-ins:")
    print("  - BM25        -> Elastic / OpenSearch")
    print("  - embed       -> sentence-transformers / OpenAI / Cohere embed")
    print("  - rerank      -> bge-reranker / Cohere rerank v3")
    print("  - generate    -> real LLM call with context in the prompt")
    print("  - metrics     -> RAGAS with an LLM judge + bootstrap CIs")


if __name__ == "__main__":
    main()
