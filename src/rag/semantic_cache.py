"""Semantic cache: short-circuit duplicate LLM calls.

Interview goal: explain how production LLM gateways save 30-70% of
inference cost by caching answers under an embedding similarity key
instead of an exact-string key.

Pattern:
  1. Embed the incoming query.
  2. Search the cache for the nearest embedding.
  3. If cosine similarity >= threshold, return the cached answer.
  4. Otherwise call the LLM, store (embedding, answer), return.

Key design choices:
  - threshold (0.92-0.97 is typical): too low -> wrong answers served,
    too high -> no cache hits.
  - eviction policy (LRU or TTL) -- we use a capped list here.
  - cache key may include model_name, temperature, and system_prompt.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


DIM = 64


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed(text: str) -> List[float]:
    v = [0.0] * DIM
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 8) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


@dataclass
class CacheEntry:
    query: str
    embedding: List[float]
    answer: str
    created_at: float


class SemanticCache:
    def __init__(self, threshold: float = 0.90, capacity: int = 1000):
        self.threshold = threshold
        self.capacity = capacity
        self.entries: List[CacheEntry] = []
        self.hits = 0
        self.misses = 0

    def lookup(self, query: str) -> Tuple[Optional[str], float]:
        q = embed(query)
        best_score, best_idx = -1.0, -1
        for i, e in enumerate(self.entries):
            s = cosine(q, e.embedding)
            if s > best_score:
                best_score, best_idx = s, i
        if best_idx >= 0 and best_score >= self.threshold:
            self.hits += 1
            return self.entries[best_idx].answer, best_score
        self.misses += 1
        return None, best_score

    def store(self, query: str, answer: str) -> None:
        self.entries.append(CacheEntry(query, embed(query), answer, time.time()))
        if len(self.entries) > self.capacity:
            self.entries.pop(0)  # FIFO eviction

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def expensive_llm(query: str) -> str:
    time.sleep(0.02)  # simulate 20ms API call
    return f"[LLM answer for: {query}]"


def cached_llm(query: str, cache: SemanticCache) -> Tuple[str, bool]:
    hit, score = cache.lookup(query)
    if hit is not None:
        return hit, True
    answer = expensive_llm(query)
    cache.store(query, answer)
    return answer, False


def main():
    cache = SemanticCache(threshold=0.85)

    traffic = [
        "What is the capital of France?",
        "Tell me the capital city of France",
        "What's France's capital?",
        "Who won the 2022 FIFA World Cup?",
        "Which country won the 2022 World Cup?",
        "What is the capital of France?",              # exact repeat
        "How tall is the Eiffel Tower?",
        "What is the capital of Germany?",
    ]

    print(f"{'#':>2}  {'hit?':<5} {'score':>7}  query")
    t0 = time.perf_counter()
    for i, q in enumerate(traffic, 1):
        ans, was_hit = cached_llm(q, cache)
        _, score = cache.lookup(q) if was_hit else (None, 0.0)
        tag = "HIT" if was_hit else "miss"
        print(f"{i:>2}  {tag:<5} {score if was_hit else 0.0:>7.3f}  {q}")
    elapsed = time.perf_counter() - t0

    print()
    print(f"Total queries : {len(traffic)}")
    print(f"Cache hits    : {cache.hits}")
    print(f"Cache misses  : {cache.misses}")
    print(f"Hit rate      : {cache.hit_rate():.0%}")
    print(f"Wall time     : {elapsed*1000:.0f} ms")
    print(f"Savings       : ~{cache.hits * 20} ms of LLM latency avoided")

    print("\nOperational guardrails:")
    print("  - Always version the cache key by model and system prompt.")
    print("  - Invalidate on new deploys; stale answers hurt user trust.")
    print("  - Log (query, nearest_query, score) for cache tuning.")
    print("  - Use conservative thresholds (>= 0.93) for safety-critical")
    print("    domains; cached wrong answers are cheap but embarrassing.")


if __name__ == "__main__":
    main()
