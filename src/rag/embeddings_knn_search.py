"""Dense retrieval baseline: hashed embeddings + cosine k-NN.

Interview goal: explain what a "vector database" actually does under
the hood. Real systems use trained embedding models (sentence-BERT,
OpenAI ada, etc.), but the *retrieval* layer is just:
  1. encode each doc to a fixed-dim vector
  2. L2-normalize so dot-product == cosine similarity
  3. compute scores against the query vector
  4. return the top-k indices

We use a deterministic hashing-trick "embedding" (no model download)
so the script runs offline. The retrieval math is identical to FAISS.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np


DIM = 64


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def hash_embed(text: str, dim: int = DIM) -> np.ndarray:
    """Sum of signed token-hash basis vectors. Cheap, deterministic, offline."""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def build_index(docs: list[str]) -> np.ndarray:
    return np.stack([hash_embed(d) for d in docs])


def search(index: np.ndarray, query: str, k: int = 3):
    q = hash_embed(query)
    scores = index @ q  # cosine similarity (rows are unit vectors)
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]


def recall_at_k(index, queries, gold_indices, k: int) -> float:
    hits = 0
    for q, gold in zip(queries, gold_indices):
        ids = [i for i, _ in search(index, q, k=k)]
        if gold in ids:
            hits += 1
    return hits / len(queries)


def main():
    docs = [
        "Transformers use self-attention to mix token representations.",
        "BERT is a bidirectional encoder pretrained with masked language modeling.",
        "GPT models are autoregressive decoders trained on next-token prediction.",
        "RAG combines a retriever and a generator to ground LLM answers.",
        "FAISS is a library for fast approximate nearest neighbor search.",
        "LoRA injects trainable low-rank matrices into frozen layers.",
        "Quantization reduces model precision to shrink memory footprint.",
        "Sentence-BERT produces dense embeddings tuned for semantic similarity.",
    ]
    index = build_index(docs)
    print(f"Index shape: {index.shape}  (n_docs, embed_dim)")

    queries = [
        ("What is FAISS used for?", 4),
        ("How do GPT models work?", 2),
        ("Tell me about low-rank fine-tuning", 5),
        ("What is masked language modeling?", 1),
    ]
    print("\nTop-3 retrieval results:")
    for q, _ in queries:
        print(f"\n  Q: {q}")
        for rank, (i, s) in enumerate(search(index, q, k=3), 1):
            print(f"    {rank}. score={s:+.3f}  {docs[i]}")

    qs = [q for q, _ in queries]
    gold = [g for _, g in queries]
    print("\nRetrieval quality:")
    for k in [1, 3, 5]:
        print(f"  Recall@{k}: {recall_at_k(index, qs, gold, k):.2f}")
    print("\nA real system would swap hash_embed for a trained encoder,")
    print("and the index for FAISS/HNSW. The math above is identical.")


if __name__ == "__main__":
    main()
