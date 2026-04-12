"""KV cache for autoregressive decoding. The speedup trick behind fast LLM inference.

Interview goal: explain why autoregressive decoding without a KV cache
is O(T^2) per step and with a KV cache is O(T) per step, and be ready
to discuss the memory cost (cache grows linearly with context length
times num_layers times num_heads times head_dim, for both K and V).

We implement:
  1. naive_decode: recompute attention over the full prefix each step.
  2. cached_decode: append new K, V and attend only with the latest Q.

Then we assert the outputs match bit-for-bit and report the FLOP ratio.
"""

from __future__ import annotations

import time

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V):
    # Q: (T_q, d), K: (T_k, d), V: (T_k, d)
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    return softmax(scores) @ V


class TinyAttentionLayer:
    def __init__(self, d: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Wq = rng.standard_normal((d, d)) * 0.1
        self.Wk = rng.standard_normal((d, d)) * 0.1
        self.Wv = rng.standard_normal((d, d)) * 0.1

    def project(self, x):
        return x @ self.Wq, x @ self.Wk, x @ self.Wv


def naive_decode(layer, tokens: np.ndarray, T_gen: int) -> np.ndarray:
    """At each generation step, recompute attention over the full prefix."""
    out_history = []
    x = tokens.copy()
    for _ in range(T_gen):
        Q, K, V = layer.project(x)
        out = attention(Q, K, V)
        next_tok = out[-1:]  # only the last row is "new"
        out_history.append(next_tok)
        x = np.concatenate([x, next_tok], axis=0)
    return np.concatenate(out_history, axis=0)


def cached_decode(layer, tokens: np.ndarray, T_gen: int) -> np.ndarray:
    """Maintain K, V caches. Per step: project ONE new token, append, attend."""
    # Prefill: project the prompt once.
    Q, K, V = layer.project(tokens)
    out_history = []

    # First generated token is attention at the last prefill position.
    out_history.append((attention(Q[-1:], K, V)))

    x_last = out_history[-1]
    for _ in range(T_gen - 1):
        q_new, k_new, v_new = layer.project(x_last)
        K = np.concatenate([K, k_new], axis=0)
        V = np.concatenate([V, v_new], axis=0)
        out = attention(q_new, K, V)  # only ONE new query vector
        out_history.append(out)
        x_last = out
    return np.concatenate(out_history, axis=0)


def main():
    rng = np.random.default_rng(1)
    d = 32
    T_prompt = 64
    T_gen = 32
    tokens = rng.standard_normal((T_prompt, d))
    layer = TinyAttentionLayer(d)

    t0 = time.perf_counter()
    y_naive = naive_decode(layer, tokens, T_gen)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_cached = cached_decode(layer, tokens, T_gen)
    t_cached = time.perf_counter() - t0

    assert y_naive.shape == y_cached.shape == (T_gen, d)
    max_diff = float(np.abs(y_naive - y_cached).max())
    print(f"Prompt length : {T_prompt}")
    print(f"Generate      : {T_gen} tokens")
    print(f"Naive  time   : {t_naive*1000:.1f} ms")
    print(f"Cached time   : {t_cached*1000:.1f} ms   ({t_naive/t_cached:.1f}x faster)")
    print(f"Max output diff: {max_diff:.2e}  (should be ~0)")
    assert max_diff < 1e-10, "KV cache must produce identical outputs"

    # Rough memory accounting (per layer, per head).
    bytes_per_token = 2 * d * 4  # K + V, float32
    cache_mb = bytes_per_token * (T_prompt + T_gen) / 1024 / 1024
    print(f"\nCache memory : ~{cache_mb:.4f} MB (1 layer, 1 head, float32)")
    print("Real models multiply by n_layers * n_heads, so KV cache is the")
    print("dominant memory cost at long context. Tricks to shrink it:")
    print("  - multi-query / grouped-query attention (share K, V across heads)")
    print("  - INT8 / FP8 cache quantization")
    print("  - paged attention (vLLM) to avoid fragmentation")


if __name__ == "__main__":
    main()
